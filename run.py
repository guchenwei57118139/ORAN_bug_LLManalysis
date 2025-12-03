from pathlib import Path
import google.genai as genai
from google.genai.types import GenerateContentConfig,Tool,FileSearch
import glob
import os, json, argparse, mimetypes
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError
from copy import deepcopy

import subprocess, re
#-----------global output file-----------

input_dict=json.load(open('input.json','r'))
output_data={}
# ---------- util func ----------

def build_global_defs(spec: dict) -> dict:
    return deepcopy(spec.get("$defs", {}))

def inject_defs_into_schema(step_schema: dict, global_defs: dict) -> dict:
    """
    combine global $defs to the current step.output_schema
    """
    merged = deepcopy(step_schema)
    local_defs = deepcopy(merged.get("$defs", {}))
    # 1st is the global, 2nd is the local
    combined = {**(global_defs or {}), **local_defs}
    if combined:
        merged["$defs"] = combined
    #merged.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    return merged
def guess_mime(p: Path):
    if p.suffix.lower() == ".md":
        return "text/markdown"
    if p.suffix.lower() in {".c", ".h", ".cpp", ".cc", ".hpp", ".txt", ".py"}:
        return "text/plain"
    mt, _ = mimetypes.guess_type(str(p))
    return mt or "text/plain"
def expand_fetch_requests(repo_root: Path, fetch_requests: list) -> list[str]:
    """
    make the control.fetch_requests from step3 into the local file path list
    support repo_root's glob, and suffix'white list and max_files constraints.
    """
    selected = []
    seen = set()
    for req in fetch_requests or []:
        pattern = req.get("pattern")
        if not pattern:
            continue
        max_files = int(req.get("max_files") or 10)
        # suffixes = set(req.get("suffix") or [])
        raw_suffixes = req.get("suffix") or []
        suffixes = set()
        for s in raw_suffixes:
            if not s:
                continue
            s = s.strip().lower()
            if not s.startswith("."):
                s = "." + s
            suffixes.add(s)
        
        # allow absolute/user directory or relative repo_root 
        if pattern.startswith(("/", "~")):
            candidates = glob.glob(os.path.expanduser(pattern), recursive=True)
        else:
            candidates = glob.glob(str((repo_root / pattern)), recursive=True)
        for p in candidates:
            if not os.path.isfile(p):
                continue
            if suffixes and Path(p).suffix not in suffixes:
                continue
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            selected.append(ap)
            if len(selected) >= max_files:
                break
    return selected
def render_prompt(template: str, inputs: dict):
    """
      - {{task}} / {{BUG_TEXT}}  filled by inputs' key with the same name
      - {{name}} / {{mime_type}} filled by spec_toc[0]'s name/mime
      - {{#each files}}... ignore
    """
    t = template
    for k in ["task", "BUG_TEXT", "bug_text"]:
        if k in inputs:
            t = t.replace("{{" + k + "}}", str(inputs[k]))
    # step2 uses {{name}} ({{mime_type}})
    if "spec_toc" in inputs and isinstance(inputs["spec_toc"], list) and inputs["spec_toc"]:
        first = inputs["spec_toc"][0]
        t = t.replace("{{name}}", str(first.get("name","file")))
        t = t.replace("{{mime_type}}", str(first.get("mime_type","application/pdf")))
    # eliminate each placeholder roughly
    t = t.replace("{{#each files}}- {{name}} ({{mime_type}})\n{{/each}}", "")
    return t

def mk_config(schema: dict, temperature=0.1):
    # compatible with different SDK fields
    kwargs = dict(
        system_instruction=(
            "You are executing one step of a JSON-only workflow. "
            "Return JSON only that matches the provided schema."
        ),
        response_mime_type="application/json",
        temperature=temperature,
    )
    try:
        return GenerateContentConfig(response_schema=schema, **kwargs)
    except TypeError:
        return GenerateContentConfig(response_json_schema=schema, **kwargs)

def llm_call(client, model, step_obj, inputs: dict, attachments_paths=None, attachment_refs=None, temperature=0.1, global_defs=None):
    prompt = render_prompt(step_obj["llm_prompt_template"], inputs)
    prepared_schema = inject_defs_into_schema(step_obj["output_schema"], global_defs or {})

    cfg = mk_config(prepared_schema, temperature=temperature)


    parts = [{
        "role": "user",
        "parts": [{"text": (
            "Execute this step and return JSON matching output_schema exactly.\n\n"
            f"inputs: {json.dumps(inputs, ensure_ascii=False)}\n\n"
            f"{prompt}"
        )}]
    }]

    # attached (optional: if step2 neds spec_toc, upload here)
    attachments = []
    for fp in attachments_paths or []:
        f = client.files.upload(file=fp, config={"mime_type": guess_mime(Path(fp))})
        attachments.append(f)
        
    if attachment_refs:
        attachments.extend(attachment_refs)

    resp = client.models.generate_content(
        model=model, config=cfg, contents=parts + attachments
    )
    return resp.text

SPEC_DOCX_PATH = os.path.join(os.path.dirname(__file__), "38473-h20.docx")

def run_spec_ingestor_simple(docx_path: str, keyword: str) -> str:
    """
    call spec_ingestor.py, return the Markdown context for the given keyword.
    """
    cmd = ["python", "spec_ingestor.py", docx_path, keyword]
    out = subprocess.check_output(cmd, text=True)
    return out  # markdown string

def build_spec_toc_from_bug(bug_card):
    """
    Retrieve all keywords from bug_card.procedure_guess,
    and for each keyword, call spec_ingestor to generate a markdown file.

    Return:
      spec_toc:    [{'name': 'slice_1_xxx.md', 'mime_type': 'text/markdown'}, ...]
      slice_paths: ['/abs/path/to/slice_1_xxx.md', ...]  # for attachment
    """
    proc_guess = bug_card.get("procedure_guess", [])
    if not isinstance(proc_guess, list) or not proc_guess:
        raise RuntimeError("step2: bug_card.procedure_guess is empty; cannot build spec slices.")

    out_dir = os.path.join(os.path.dirname(__file__), "spec_slices")
    os.makedirs(out_dir, exist_ok=True)

    spec_toc = []
    slice_paths = []

    for i, kw in enumerate(proc_guess, start=1):
        kw_str = str(kw)
        # 1) call spec_ingestor to obtain markdown
        md = run_spec_ingestor_simple(SPEC_DOCX_PATH, kw_str)
        # 2) turn keyword into safe file names
        safe_kw = re.sub(r"[^A-Za-z0-9]+", "_", kw_str).strip("_") or f"kw{i}"
        file_name = f"slice_{i}_{safe_kw}.md"
        file_path = os.path.join(out_dir, file_name)
        # 3) write .md 
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        # 4) record metadata path
        slice_paths.append(file_path)
        spec_toc.append({
            "name": file_name,
            "mime_type": "text/markdown",
        })
    return spec_toc, slice_paths
# def locate_repo_root() -> Path:
#     """
#     locate openairinterface5g 仓库根目录；找不到就退回到当前目录。
#     """
#     here = Path(__file__).resolve().parent
#     for p in [here, *here.parents]:
#         cand = p / "openairinterface5g"
#         if cand.exists():
#             return cand
#     return here
def collect_code_paths(max_files: int = 80):
    project_root = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(project_root, "openairinterface5g", "openair2", "F1AP")

    if not os.path.isdir(target_dir):
        print(f"[WARN] Cannot find code directory: {target_dir}")
        return []

    code_paths = []
    for dirpath, dirnames, filenames in os.walk(target_dir):
        for fname in filenames:
            if fname.endswith((".c", ".h", ".cpp", ".cc", ".hpp")):
                full_path = os.path.join(dirpath, fname)
                code_paths.append(full_path)
                if len(code_paths) >= max_files:
                    return code_paths

    return code_paths
def write_code_tree_file(
    repo_root: Path | str,
    out_path: Path | None = None,
    include_suffixes=(".c", ".h", ".cpp", ".cc", ".hpp", ".py", ".md", ".txt"),
) -> str:
    """
    generate code tree.txt
    """
    repo_root = Path(repo_root).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")

    if out_path is None:
        # out_path = repo_root / "__code_tree.txt"
        out_path = repo_root.parent / f"{repo_root.name}__code_tree.txt"

    sufset = {s.lower() for s in include_suffixes} if include_suffixes else None

    lines = []
    for root, _, files in os.walk(repo_root):
        for f in files:
            if sufset and Path(f).suffix.lower() not in sufset:
                continue
            rel = os.path.relpath(os.path.join(root, f), repo_root)
            lines.append(rel.replace("\\", "/"))  

    lines.sort()
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
def upload_code_files_once(client, code_paths):
    uploaded = []
    for fp in code_paths:
        f = client.files.upload(file=fp, config={"mime_type": "text/plain"})
        uploaded.append(f)
    return uploaded
def upload_paths_return_refs(client, paths: list[str]):
    """
    upload local code files in batch
    return the refs list 
    """
    refs = []
    for fp in paths or []:
        try:
            f = client.files.upload(file=fp, config={"mime_type": guess_mime(Path(fp))})
            refs.append(f)
        except Exception as e:
            print(f"[WARN] upload failed: {fp}: {e}")
    return refs
def validate_json(schema, data_text):
    data = json.loads(data_text)
    Draft202012Validator(schema).validate(data)
    return data
def build_inputs_for_step(spec, ctx, step_idx, externals):
    """
    Based on the input_schema dependency in QAprompt.json, combine the inputs for each step.
    ctx: reserve the output of each step
    externals: external input (like step2.spec_toc、step3.repo_metadata)
    """
    steps = spec["steps"]

    if step_idx == 0:  # step 1: Bug Ingest
        # bug_text--external input
        return {"bug_text": externals.get("bug_text", "Describe the bug here.")}

    if step_idx == 1:  # step 2: Spec Section Fetcher
        if "step1" not in ctx:
            raise RuntimeError("step2 depends on step1's bug_card")

        bug_card = ctx["step1"]["bug_card"]
        spec_toc, slice_paths = build_spec_toc_from_bug(bug_card)
        externals["spec_slice_paths"] = slice_paths
        return {
            "bug_card": bug_card,
            "spec_toc": spec_toc,
        }


    if step_idx == 2:  # step 3: Code Fetcher
        if "step1" not in ctx or "step2" not in ctx:
            raise RuntimeError("step3 depends on step1.bug_card and step2.candidate_spec_sections")
        csecs = ctx["step2"]["candidate_spec_sections"]
        keywords = []
        for it in csecs:
            kws = it.get("Keywords", [])
            if isinstance(kws, list):
                keywords.extend([str(x) for x in kws])
        repo_md = {
            "path": spec["$defs"]["RepoEnum"]["enum"][0],
            "interface_names": ctx["step1"]["bug_card"]["interface_guess"],
            "known_keywords": list(dict.fromkeys(keywords))  
        }
        '''add iteration and max_iters'''
        step3 = spec["steps"][2]
        props = step3["input_schema"]["properties"]
        it_prop  = props.get("iteration",  {})
        max_prop = props.get("max_iters",  {})
        default_iter = it_prop.get("default", 0)
        default_max  = max_prop.get("default", 3)
        return {
            "bug_card": ctx["step1"]["bug_card"],
            "candidate_spec_sections": csecs,
            "repo_metadata": repo_md,
            "iteration": default_iter, 
            "max_iters": default_max
        }

    if step_idx == 3:  # step 4: Retriever
        if "step2" not in ctx or "step3" not in ctx:
            raise RuntimeError("step4 depends on step2.candidate_spec_sections and step3.candidate_code")
        return {
            "candidate_spec_sections": ctx["step2"]["candidate_spec_sections"],
            "candidate_code": ctx["step3"]["candidate_code"]
        }

    if step_idx == 4:  # step 5: Sketcher
        if "step1" not in ctx or "step4" not in ctx:
            raise RuntimeError("step5 depends on step1.bug_card and step4.snippets")
        return {
            "bug_card": ctx["step1"]["bug_card"],
            "snippets": ctx["step4"]["snippets"]
        }

    if step_idx == 5:  # step 6: Conformance Checker
        if "step1" not in ctx or "step4" not in ctx or "step5" not in ctx:
            raise RuntimeError("step6 depends on step1.bug_card,step5.sequence/state and step4.snippets")
        return {
            "bug_card": ctx["step1"]["bug_card"],
            "sequence_diagram": ctx["step5"]["sequence_diagram"],
            "state_machine": ctx["step5"]["state_machine"],
            "snippets": ctx["step4"]["snippets"]
        }

    if step_idx == 6:  # step 7: Evidence Linker
        if "step1" not in ctx or "step4" not in ctx or "step6" not in ctx:
            raise RuntimeError("step7 depends on step1.bug_card,step6.invariant_checks and step4.snippets")
        return {
            "bug_card": ctx["step1"]["bug_card"],
            "invariant_checks": ctx["step6"]["invariant_checks"],
            "snippets": ctx["step4"]["snippets"]
        }
    if step_idx==7:
        if "step1" not in ctx or "step6" not in ctx or "step7" not in ctx:
            raise RuntimeError("step8 depends on step1.bug_card,step6.invariant_checks and step7.evidence_pack")
        return {
            "bug_card": ctx["step1"]["bug_card"],
            "invariant_checks": ctx["step6"]["invariant_checks"],
            "evidence_pack": ctx["step7"]["evidence_pack"]
        }
    raise RuntimeError(f"Unachieved step: {step_idx+1}")



def run_pipeline(spec_path: str, model: str, client, externals: dict, temperature=0.1):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    # client = genai.Client(api_key="")
    #client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    ctx = {}
    

    global_defs = build_global_defs(spec)


    global_defs=spec.get("$defs",{})
    # step1-8
    for idx in range(8):
        step = spec["steps"][idx]
        step_name = step["step"]
        inputs = build_inputs_for_step(spec, ctx, idx, externals)

        if idx == 1:
            attach_paths = externals.get("spec_slice_paths", [])
            attach_refs = None
        elif idx == 2:
            # 1) upload code tree.txt
            repo_root = Path("./openairinterface5g").expanduser().resolve()
            code_tree_path = './openairinterface5g__code_tree.txt'
            code_tree_ref = client.files.upload(file=code_tree_path, config={"mime_type": "text/plain"})

            # 2) read ieteration params
            max_iters = int(inputs.get("max_iters", 3))
            iteration = int(inputs.get("iteration", 0))

            # initially, only code_tree.txt
            dynamic_refs = [code_tree_ref]

            last_raw_text = None
            while True:
                loop_inputs = {**inputs, "iteration": iteration}

                last_raw_text = llm_call(
                    client, model, step, loop_inputs,
                    attachments_paths=None,
                    attachment_refs=dynamic_refs,
                    temperature=temperature,
                    global_defs=global_defs,
                )

                try:
                    parsed = json.loads(last_raw_text)
                except Exception as e:
                    print(f"[WARN] step3 JSON parse failed at iter {iteration}: {e}")
                    break

                control = (parsed.get("control") or {})
                decision = (control.get("decision") or "stop").lower()
                fetch_requests = control.get("fetch_requests") or []

                if decision == "continue" and iteration + 1 < max_iters and fetch_requests:
                    new_paths = expand_fetch_requests(repo_root, fetch_requests)
                    if not new_paths:
                        print(f"[INFO] step3 iter {iteration}: no files matched fetch_requests; stopping.")
                        break
                    new_refs = upload_paths_return_refs(client, new_paths)
                    if not new_refs:
                        print(f"[INFO] step3 iter {iteration}: upload none; stopping.")
                        break
                    dynamic_refs.extend(new_refs)
                    iteration += 1
                    # input("step3 fetch continue ...")
                    continue
                else:
                    break

            prepared_schema = inject_defs_into_schema(step["output_schema"], global_defs)
            try:
                data = validate_json(prepared_schema, last_raw_text)
            except ValidationError as e:
                raise SystemExit(f"[SchemaError] step {idx+1} ({step_name}) invalid output: {e.message}\nreturn origin:\n{last_raw_text}")

            ctx[f"step{idx+1}"] = data
            output_data[f"step{idx+1}"] = data
            print(data)
            continue
        elif idx == 3:  
            attach_paths = None
            attach_refs = externals.get("code_files", [])
        else:
            attach_paths = None
            attach_refs = None

        raw = llm_call(
            client,
            model,
            step,
            inputs,
            attachments_paths=attach_paths,
            attachment_refs=attach_refs,
            temperature=temperature,
            global_defs=global_defs,
        )
        prepared_schema = inject_defs_into_schema(step["output_schema"], global_defs)

        try:
            data = validate_json(prepared_schema, raw)
        except ValidationError as e:
            raise SystemExit(f"[SchemaError] step {idx+1} ({step_name}) invalid output: {e.message}\nreturn origin:\n{raw}")

        ctx[f"step{idx+1}"] = data
        output_data[f'step{idx+1}']=data
        print(data)
    return ctx
if __name__ == "__main__":
    '''generate code tree, only run once'''
    # write_code_tree_file(repo_root='./openairinterface5g')
    '''main func'''
    client = genai.Client(api_key="AIzaSyBaqSqxBCfSsomsMGKLNISFvJU_h1Mwom0")
    code_paths = collect_code_paths()
    # code_files = upload_code_files_once(client, code_paths)
    # print(f"[INFO] collected {len(code_paths)} code files for step3")
    for bug,values in input_dict.items():
        bug_text=values['input']
        output_data={}
        fw=open(values['output'],'w')
        externals = {
            "bug_text": bug_text,
            "code_files": []
            # "code_files": [code_files]
        }    
       # externals["spec_toc"]=[{'name':'F1AP.md',"mime_type":'text/markdown'}]
        ctx = run_pipeline(
            spec_path='QAprompt.json',
            model="gemini-2.5-flash",
            client=client,  
            externals=externals,
            temperature=0.1
        )
        print(f"========={bug} report OK=========")
        json.dump(output_data,fw,indent=2)
        


