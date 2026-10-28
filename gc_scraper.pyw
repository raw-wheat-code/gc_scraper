import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

ROOT_WIDTH = 600
ROOT_HEIGHT = 500

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MAIN_WORKFLOW_SCRIPT = os.path.join(BASE_DIR, "py", "main_workflow.py")
SET_TOKEN_SCRIPT = os.path.join(BASE_DIR, "py", "set_token.py")
GET_TOKEN_SCRIPT = os.path.join(BASE_DIR, "py", "get_token.py")
TOKEN_FILE_PATH = os.path.join(BASE_DIR, "config", "token.txt")


def run_workflow():
    schedule_url = url_var.get().strip()
    if not schedule_url:
        messagebox.showerror("Missing URL", "Please enter a team schedule URL.")
        return

    token_arg = "true" if run_token_var.get() else "false"
    command = ["python", MAIN_WORKFLOW_SCRIPT, schedule_url, token_arg]

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        messagebox.showinfo("Success", "Pipeline completed successfully.")
    except subprocess.CalledProcessError as exc:
        messagebox.showerror(
            "Pipeline failed",
            f"Command exited with code {exc.returncode}.\n\nStdout:\n{exc.stdout.strip()}\n\nStderr:\n{exc.stderr.strip()}"
        )
    except Exception as exc:
        messagebox.showerror("Error", str(exc))


def toggle_token_panel():
    if token_frame.winfo_ismapped():
        token_frame.grid_remove()
    else:
        token_frame.grid()


def set_token():
    token_text = token_var.get().strip()
    if not token_text:
        messagebox.showerror("Missing Token", "Please enter a token before setting it.")
        return

    if token_text.startswith("Bearer "):
        token_text = token_text[7:].strip()

    command = ["python", SET_TOKEN_SCRIPT, token_text]

    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        messagebox.showinfo("Token Updated", "GC token updated successfully.")
    except subprocess.CalledProcessError as exc:
        messagebox.showerror(
            "Token failed",
            f"Command exited with code {exc.returncode}.\n\nStdout:\n{exc.stdout.strip()}\n\nStderr:\n{exc.stderr.strip()}"
        )
    except Exception as exc:
        messagebox.showerror("Error", str(exc))


def get_token():
    try:
        completed = subprocess.run(
            ["python", GET_TOKEN_SCRIPT],
            check=True,
            capture_output=True,
            text=True
        )

        if os.path.exists(TOKEN_FILE_PATH):
            with open(TOKEN_FILE_PATH, "r", encoding="utf-8") as f:
                extracted_token = f.read().strip()
                if extracted_token:
                    token_var.set(extracted_token)

        messagebox.showinfo("Token Extracted", "GC token extracted successfully.")
    except subprocess.CalledProcessError as exc:
        messagebox.showerror(
            "Get Token Failed",
            f"Command exited with code {exc.returncode}.\n\nStdout:\n{exc.stdout.strip()}\n\nStderr:\n{exc.stderr.strip()}"
        )
    except Exception as exc:
        messagebox.showerror("Error", str(exc))


def build_ui(root):
    root.title("GC Scraper")
    root.geometry(f"{ROOT_WIDTH}x{ROOT_HEIGHT}")
    root.resizable(False, False)

    main_frame = tk.Frame(root, padx=16, pady=16)
    main_frame.pack(fill="both", expand=True)

    tk.Label(main_frame, text="GC Scraper", font=("Segoe UI", 20, "bold")).pack(pady=(0, 16))

    tk.Label(main_frame, text="Team Schedule URL:", anchor="w").pack(fill="x")
    tk.Entry(main_frame, textvariable=url_var, width=60).pack(fill="x", pady=(4, 12))

    tk.Checkbutton(main_frame, text="Run token extractor", variable=run_token_var).pack(anchor="w", pady=(0, 12))

    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill="x", pady=(0, 8))

    tk.Button(button_frame, text="Run", width=16, command=run_workflow).pack(side="left")
    tk.Button(button_frame, text="Token", width=16, command=toggle_token_panel).pack(side="left", padx=(12, 0))

    global token_frame
    token_frame = tk.Frame(main_frame, pady=12)
    token_frame.pack(fill="x")

    tk.Label(token_frame, text="New GC Token:", anchor="w").grid(row=0, column=0, sticky="w")
    tk.Entry(token_frame, textvariable=token_var, width=60).grid(row=1, column=0, sticky="we", pady=(4, 8), columnspan=2)
    tk.Button(token_frame, text="Get", width=16, command=get_token).grid(row=2, column=0, sticky="w")
    tk.Button(token_frame, text="Set", width=16, command=set_token).grid(row=2, column=1, sticky="w")

    token_frame.grid_remove()


if __name__ == "__main__":
    root = tk.Tk()

    url_var = tk.StringVar()
    run_token_var = tk.BooleanVar(value=False)
    token_var = tk.StringVar()

    build_ui(root)
    root.mainloop()