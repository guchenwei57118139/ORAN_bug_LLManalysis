from pathlib import Path
import google.genai as genai
from google.genai.types import GenerateContentConfig,Tool,FileSearch

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
    mt, _ = mimetypes.guess_type(str(p))
    return mt or "text/plain"

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

def llm_call(client, model, step_obj, inputs: dict, attachments_paths=None, temperature=0.1, global_defs=None):
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

    resp = client.models.generate_content(
        model=model, config=cfg, contents=parts + attachments
    )
    return resp.text

SPEC_DOCX_PATH = os.path.join(os.path.dirname(__file__), "38473-h20.docx")

def run_spec_ingestor_simple(docx_path: str, keyword: str) -> str:
    """
    调用 spec_ingestor.py，返回给定 keyword 的 Markdown 上下文。
    """
    cmd = ["python", "spec_ingestor.py", docx_path, keyword]
    out = subprocess.check_output(cmd, text=True)
    return out  # markdown string

def build_spec_toc_from_bug(bug_card):
    """
    从 bug_card.procedure_guess 里取出所有 keyword，
    对每个 keyword 调用 spec_ingestor，生成一个 markdown 文件。

    返回:
      spec_toc:    [{'name': 'slice_1_xxx.md', 'mime_type': 'text/markdown'}, ...]
      slice_paths: ['/abs/path/to/slice_1_xxx.md', ...]  # 给附件上传用
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
        # 1) 调 spec_ingestor 拿到 markdown
        md = run_spec_ingestor_simple(SPEC_DOCX_PATH, kw_str)
        # 2) 把 keyword 变成安全一点的文件名
        safe_kw = re.sub(r"[^A-Za-z0-9]+", "_", kw_str).strip("_") or f"kw{i}"
        file_name = f"slice_{i}_{safe_kw}.md"
        file_path = os.path.join(out_dir, file_name)
        # 3) 写入 .md 文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md)
        # 4) 记录 metadata 和路径
        slice_paths.append(file_path)
        spec_toc.append({
            "name": file_name,
            "mime_type": "text/markdown",
        })
    return spec_toc, slice_paths

def validate_json(schema, data_text):
    data = json.loads(data_text)
    Draft202012Validator(schema).validate(data)
    return data
def build_inputs_for_step(spec, ctx, step_idx, externals):
    """
    根据 QAprompt.json 的 input_schema 依赖，组合每一步的 inputs。
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
        return {
            "bug_card": ctx["step1"]["bug_card"],
            "candidate_spec_sections": csecs,
            "repo_metadata": repo_md
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



def run_pipeline(spec_path: str, model: str, externals: dict, attach_files=None, temperature=0.1):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    client=genai.Client(api_key="")
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
            attach = externals.get("spec_slice_paths", [])
        else:
            attach = None

        raw = llm_call(client, model, step, inputs, attachments_paths=attach, temperature=temperature,global_defs=global_defs, )
        prepared_schema = inject_defs_into_schema(step["output_schema"], global_defs)

        try:
            data = validate_json(prepared_schema, raw)
        except ValidationError as e:
            raise SystemExit(f"[SchemaError] step {idx+1} ({step_name}) invalid output: {e.message}\nreturn origin:\n{raw}")

        ctx[f"step{idx+1}"] = data
        output_data[f'step{idx+1}']=data

    return ctx
if __name__ == "__main__":
    for bug,values in input_dict.items():
        if bug=='bug1' or bug=='bug2' or bug=='bug3' or bug=='bug4' or bug=='bug5':continue
        bug_text=values['input']
        output_data={}
        fw=open(values['output'],'w')
        externals={'bug_text':bug_text}    
       # externals["spec_toc"]=[{'name':'F1AP.md',"mime_type":'text/markdown'}]
        ctx = run_pipeline(
            spec_path='QAprompt.json',
            model="gemini-2.5-flash",
            externals=externals,
            attach_files=None,
            temperature=0.1
        )
        print(f"========={bug} report OK=========")
        json.dump(output_data,fw,indent=2)
