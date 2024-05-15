import llvm

def detect_loop(llvm_ir):
  """Detects a loop in LLVM IR.

  Args:
    llvm_ir: The LLVM IR to detect a loop in.

  Returns:
    True if a loop is detected, False otherwise.
  """

  loop_info = llvm.LoopInfo(llvm_ir)
  for loop in loop_info.get_loops():
    for instruction in loop.get_instructions():
      if instruction.get_function() == llvm_ir:
        return True

  return False

# Example usage:

llvm_ir = """
define i32 @main() {
  %i = alloca i32
  store i32 0, %i
  br label %loop

loop:
  %j = load i32, %i
  %inc = add i32 %j, 1
  store i32 %inc, %i
  br i1 icmp ne i32 %inc, 10, label %loop, label %end

end:
  ret i32 0
}
"""

if detect_loop(llvm_ir):
  print("A loop was detected in the LLVM IR.")
else:
  print("No loop was detected in the LLVM IR.")