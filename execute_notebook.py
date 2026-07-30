import sys
import os
import io
import json
import base64
import ast
import matplotlib.pyplot as plt

def exec_with_last_expr(code_str, glob):
    # Parse the code into an AST
    tree = ast.parse(code_str)
    
    # If the last statement is an expression, we split the AST
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1]
        # Create a new AST with everything except the last expression
        tree.body = tree.body[:-1]
        # Execute the main block
        if tree.body:
            exec(compile(tree, filename="<ast>", mode="exec"), glob)
        # Evaluate the last expression
        val = eval(compile(ast.Expression(body=last_expr.value), filename="<ast>", mode="eval"), glob)
        return val
    else:
        exec(compile(tree, filename="<ast>", mode="exec"), glob)
        return None

def run():
    print("Loading Assignment-8.ipynb...")
    with open("Assignment-8.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    # Global execution namespace
    glob = {}
    
    # We will monkeypatch plt.show to capture the current figure, convert it to base64, and store it.
    current_plot_base64 = []
    
    original_show = plt.show
    def mock_show(*args, **kwargs):
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        current_plot_base64.append(img_b64)
        plt.close() # Close current figure
    
    plt.show = mock_show
    
    cell_idx = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell_idx += 1
            print(f"Executing code cell {cell_idx}...")
            
            # Clear previous outputs
            cell["outputs"] = []
            cell["execution_count"] = cell_idx
            
            # Join the source lines to execute
            code = "".join(cell["source"])
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            current_plot_base64.clear()
            
            try:
                # Run code with last expression evaluation
                val = exec_with_last_expr(code, glob)
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print(f"Error in cell {cell_idx}: {e}")
                # Append error to outputs
                cell["outputs"].append({
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)]
                })
                continue
            
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
            
            # Add stdout stream output
            if stdout_text:
                cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout_text.splitlines(keepends=True)
                })
            
            # Add stderr stream output
            if stderr_text:
                cell["outputs"].append({
                    "output_type": "stream",
                    "name": "stderr",
                    "text": stderr_text.splitlines(keepends=True)
                })
            
            # Add plot display data outputs
            for img_b64 in current_plot_base64:
                cell["outputs"].append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {}
                })
            
            # Add last expression evaluation result
            if val is not None:
                data = {"text/plain": [repr(val)]}
                if hasattr(val, "_repr_html_"):
                    data["text/html"] = [val._repr_html_()]
                cell["outputs"].append({
                    "output_type": "execute_result",
                    "execution_count": cell_idx,
                    "data": data,
                    "metadata": {}
                })
    
    # Restore show
    plt.show = original_show
    
    # Save notebook back
    with open("Assignment-8.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Notebook executed and saved successfully!")

if __name__ == "__main__":
    run()
