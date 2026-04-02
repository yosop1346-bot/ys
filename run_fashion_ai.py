import json
import os
import shutil
import subprocess
import tempfile
import threading
import webbrowser
import ast
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request


BASE_DIR = Path(__file__).resolve().parent
HTML_FILE = "fashion-ai-v3.html"
PORT = int(os.getenv("PORT", "8080"))
HOST = os.getenv("HOST", "0.0.0.0")
AUTO_OPEN_BROWSER = os.getenv("AUTO_OPEN_BROWSER", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
RUN_TIMEOUT_SECONDS = int(os.getenv("PYTHON_RUN_TIMEOUT_SECONDS", "3"))
MAX_OUTPUT_CHARS = int(os.getenv("PYTHON_RUN_MAX_OUTPUT_CHARS", "4000"))
MAX_CODE_LENGTH = int(os.getenv("PYTHON_RUN_MAX_CODE_LENGTH", "8000"))


def build_system_prompt(answer_style: str = "") -> str:
    style_guidance = {
        "balanced": (
            "Choose the clearest response style for the question. "
            "You may answer briefly or in more detail depending on what helps most."
        ),
        "simple": (
            "Use very simple wording, short sentences, and beginner-friendly explanations."
        ),
        "example": (
            "Favor examples and short code snippets when they would help understanding."
        ),
        "concise": (
            "Keep answers compact and direct unless the user clearly asks for detail."
        ),
        "deep": (
            "Explain more deeply, include reasoning, and compare options when useful."
        ),
    }
    selected_style = style_guidance.get(answer_style, style_guidance["balanced"])

    return (
        "You are a friendly coding tutor for a beginner. "
        "Adapt your answer style to the user's intent instead of using one fixed format. "
        "You can answer both problem-specific questions and general beginner Python questions. "
        "Examples of valid general questions: what def means, what return does, what a variable is, what a list is. "
        "If the question is simple, answer simply. "
        "If the user seems confused, slow down and explain step by step. "
        "If examples would help, include them. "
        "If multiple interpretations are possible, choose the most helpful one and clarify briefly. "
        "Point out mistakes gently and give practical next steps. "
        "When code context is provided, use it in your explanation. "
        "Prefer Korean unless the user asks for English. "
        f"Style preference: {selected_style}"
    )


def call_openai(messages):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is missing. "
            "Set it before starting the app."
        )

    payload = {
        "model": DEFAULT_MODEL,
        "input": messages,
        "temperature": DEFAULT_TEMPERATURE,
    }

    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error while calling OpenAI: {exc.reason}") from exc

    output_text = data.get("output_text")
    if output_text:
        return output_text.strip()

    # Fallback for unexpected response shapes.
    collected = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                collected.append(text)
    return "\n".join(collected).strip()


def get_python_command():
    for candidate in ("python", "py"):
        path = shutil.which(candidate)
        if path:
            if candidate == "py":
                return [path, "-3", "-I"]
            return [path, "-I"]
    return None


def validate_python_code(source: str):
    if not source.strip():
        return "실행할 코드가 비어 있습니다."

    if len(source) > MAX_CODE_LENGTH:
        return f"코드가 너무 깁니다. {MAX_CODE_LENGTH}자 이하로 줄여 주세요."

    blocked_substrings = [
        "subprocess",
        "os.system",
        "shutil.rmtree",
        "Path.unlink",
        "Path.rmdir",
        "input(",
        "open(",
        "__import__",
        "eval(",
        "exec(",
        "compile(",
    ]
    lowered = source.lower()
    for token in blocked_substrings:
        if token.lower() in lowered:
            return f"안전상의 이유로 `{token}` 사용은 허용되지 않습니다."

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"문법 오류가 있습니다: {exc.msg} (line {exc.lineno})"

    blocked_imports = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "ctypes",
        "signal",
        "asyncio",
        "threading",
        "multiprocessing",
        "urllib",
        "http",
        "requests",
        "builtins",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in blocked_imports:
                    return f"안전상의 이유로 `{top}` 모듈 import는 허용되지 않습니다."
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in blocked_imports:
                return f"안전상의 이유로 `{module}` 모듈 import는 허용되지 않습니다."
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "open",
                "exec",
                "eval",
                "compile",
                "input",
                "__import__",
            }:
                return f"안전상의 이유로 `{node.func.id}()` 사용은 허용되지 않습니다."

    return None


def truncate_output(text: str):
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    clipped = text[:MAX_OUTPUT_CHARS]
    return clipped + f"\n\n[출력이 길어서 {MAX_OUTPUT_CHARS}자까지만 표시했습니다.]"


def run_python_code(source: str):
    validation_error = validate_python_code(source)
    if validation_error:
        return {
            "ok": False,
            "stdout": "",
            "stderr": validation_error,
            "timed_out": False,
        }

    command = get_python_command()
    if not command:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "Python 실행기를 찾지 못했습니다. Python 설치를 확인해 주세요.",
            "timed_out": False,
        }

    with tempfile.TemporaryDirectory(prefix="fashion_ai_run_") as temp_dir:
        script_path = Path(temp_dir) / "student_code.py"
        script_path.write_text(source, encoding="utf-8")

        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }

        try:
            completed = subprocess.run(
                command + [str(script_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=temp_dir,
                env=env,
                timeout=RUN_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = truncate_output(exc.stdout or "")
            stderr = truncate_output((exc.stderr or "") + "\n실행 시간이 너무 길어서 중단했습니다.")
            return {
                "ok": False,
                "stdout": stdout,
                "stderr": stderr.strip(),
                "timed_out": True,
            }
        except OSError as exc:
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"실행 중 시스템 오류가 발생했습니다: {exc}",
                "timed_out": False,
            }

    stdout = truncate_output(completed.stdout or "")
    stderr = truncate_output(completed.stderr or "")
    return {
        "ok": completed.returncode == 0,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
    }


class CodingAssistantHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def guess_type(self, path):
        guessed = super().guess_type(path)
        if guessed == "text/html":
            return "text/html; charset=utf-8"
        if guessed == "application/json":
            return "application/json; charset=utf-8"
        if guessed.startswith("text/") and "charset=" not in guessed:
            return f"{guessed}; charset=utf-8"
        return guessed

    def do_POST(self):
        if self.path == "/api/run":
            self._handle_run()
            return

        if self.path != "/api/chat":
            self.send_error(404, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
            return

        question = (payload.get("question") or "").strip()
        code_context = (payload.get("codeContext") or "").strip()
        learning_goal = (payload.get("learningGoal") or "").strip()
        history = payload.get("history") or []
        answer_style = (payload.get("answerStyle") or "balanced").strip().lower()

        if not question:
            self._send_json({"error": "Question is required."}, status=400)
            return

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": build_system_prompt(answer_style)}
                ],
            }
        ]

        if learning_goal:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"My current learning goal is: {learning_goal}",
                        }
                    ],
                }
            )

        if answer_style:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"My preferred answer style is: {answer_style}",
                        }
                    ],
                }
            )

        if code_context:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Here is the code or note I am currently studying:\n"
                            + code_context,
                        }
                    ],
                }
            )

        for item in history[-8:]:
            role = item.get("role")
            text = (item.get("text") or "").strip()
            if role in {"user", "assistant"} and text:
                messages.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": question}],
            }
        )

        try:
            answer = call_openai(messages)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        self._send_json({"answer": answer, "model": DEFAULT_MODEL})

    def _handle_run(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
            return

        source = payload.get("code") or ""
        result = run_python_code(source)
        status = 200 if result["ok"] else 400
        self._send_json(result, status=status)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_server():
    server = ThreadingHTTPServer((HOST, PORT), CodingAssistantHandler)
    local_url = f"http://127.0.0.1:{PORT}/{HTML_FILE}"
    public_hint = f"http://{HOST}:{PORT}/{HTML_FILE}"
    print(f"Coding assistant is running at {local_url}")
    if HOST != "127.0.0.1":
        print(f"Server binding: {public_hint}")
    print("Press Ctrl+C in this terminal to stop the server.")
    server.serve_forever()


if __name__ == "__main__":
    should_open_browser = AUTO_OPEN_BROWSER and PORT == 8080

    if should_open_browser:
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        webbrowser.open(f"http://127.0.0.1:{PORT}/{HTML_FILE}")

        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\nServer stopped.")
    else:
        try:
            start_server()
        except KeyboardInterrupt:
            print("\nServer stopped.")
