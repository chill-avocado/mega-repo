# Common code_executor component for code_execution
class CodeExecutor:
    """Safe code execution environment."""
    
    def __init__(self, timeout=30):
        """
        Initialize the code executor.
        
        Args:
            timeout (int): Maximum execution time in seconds.
        """
        self.timeout = timeout
        self.globals = {}
        self.locals = {}
    
    def execute(self, code):
        """
        Execute Python code safely.
        
        Args:
            code (str): Python code to execute.
        
        Returns:
            dict: Execution results with stdout, stderr, and return value.
        """
        import sys
        import io
        import traceback
        import threading
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        
        # Prepare result
        result = {
            'stdout': '',
            'stderr': '',
            'return_value': None,
            'error': None,
            'timed_out': False
        }
        
        # Create execution function
        def exec_code():
            sys_stdout = sys.stdout
            sys_stderr = sys.stderr
            
            try:
                sys.stdout = stdout
                sys.stderr = stderr
                
                # Execute the code
                exec(code, self.globals, self.locals)
                
                # Get the last expression's value if it exists
                if '_' in self.locals:
                    result['return_value'] = self.locals['_']
            except Exception as e:
                result['error'] = str(e)
                result['stderr'] = traceback.format_exc()
            finally:
                sys.stdout = sys_stdout
                sys.stderr = sys_stderr
        
        # Execute with timeout
        thread = threading.Thread(target=exec_code)
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            # Timeout occurred
            result['timed_out'] = True
            result['stderr'] = f"Execution timed out after {self.timeout} seconds"
            
            # Try to terminate the thread (not guaranteed)
            try:
                import ctypes
                thread_id = thread.ident
                if thread_id:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id),
                        ctypes.py_object(SystemExit)
                    )
            except:
                pass
        else:
            # Execution completed
            result['stdout'] = stdout.getvalue()
            result['stderr'] = stderr.getvalue()
        
        return result
