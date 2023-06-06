# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import jaxlib.mlir.ir as mlir_ir
import jax._src.interpreters.mlir as mlir
import multiprocessing
import os
import re

parser = argparse.ArgumentParser(prog='triage_jaxtest.py',
                                 description='Triage the jax tests')
parser.add_argument('-l', '--logdir', default="/tmp/jaxtest")
parser.add_argument('-d', '--delete', default=False)
parser.add_argument('-j', '--jobs', default=None)
args = parser.parse_args()

tests = set(os.listdir(args.logdir))


def filter_to_failures(tests):
  failures = list()
  for test in tests:
    files = os.listdir(f"{args.logdir}/{test}")
    if "error.txt" in files or "CRASH_MARKER" in files:
      failures.append(test)
  failures = sorted(failures)
  return failures


def check_custom_call(errortxt, mlirbc, __):
  return "stablehlo.custom_call" in errortxt or "stablehlo.custom_call" in mlirbc


def check_load_ui(errortxt, _, __):
  return "'flow.tensor.load' op result #0 must be index or signless integer or floating-point or complex-type or vector of any type values, but got 'ui32'" in errortxt


def check_splat_ui(errortxt, _, __):
  return "'flow.tensor.splat' op failed to verify that value type matches element type of result" in errortxt and "xui" in errortxt


def check_degenerate_tensor(_, mlirbc, __):
  return "tensor<0x" in mlirbc or "x0x" in mlirbc


def check_topk_bf16(_, mlirbc, __):
  return "bf16" in mlirbc and "hlo.top_k" in mlirbc


def check_cross_replica(errortxt, _, __):
  return "cross-replica" in errortxt


def check_collective(errortxt, _, __):
  return "stablehlo.collective" in errortxt or "UNIMPLEMENTED; collectives not implemented" in errortxt


def check_optimization_barrier(errortxt, _, __):
  return "util.optimization_barrier" in errortxt


def check_sort_shape(errortxt, _, __):
  return "'iree_linalg_ext.sort' op expected operand 1 to have same shape as other operands" in errortxt


def check_reverse_i1(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    if "stablehlo.reverse" in line and "xui" in line:
      return True
  return False


def check_complex(errortxt, mlirbc, __):
  return "complex<" in errortxt or "complex<" in mlirbc


def check_timeout(errortxt, _, __):
  return "jaxlib.xla_extension.XlaRuntimeError: ABORTED: ABORTED" in errortxt


def check_rng_bit(errortxt, _, __):
  return "stablehlo.rng_bit_generator" in errortxt


def check_min_max_f16(errortxt, _, __):
  lines = errortxt.split("\n")
  for line in lines:
    has_fmax = "llvm.intr.vector.reduce.fmax" in line
    has_fmin = "llvm.intr.vector.reduce.fmin" in line
    has_f16 = "f16" in line
    if (has_fmax or has_fmin) and has_f16:
      return True
  return False


def check_scatter_ui(errortxt, _, __):
  lines = errortxt.split("\n")
  for line in lines:
    has_scatter = "iree_linalg_ext.scatter" in line
    has_operand = "expected type of `outs` operand #0" in line
    has_type = "xui" in line
    if has_scatter and has_operand and has_type:
      return True
  return False


def check_bitcast_bf16(errortxt, _, __):
  return "bf16" in errortxt and "`arith.bitcast` op operand type" in errortxt


def check_constant_bf16(errortxt, _, __):
  return "FloatAttr does not match expected type of the constant" in errortxt


def check_triangular_solve(errortxt, _, __):
  return "stablehlo.triangular_solve" in errortxt


def check_cholesky(errortxt, _, __):
  return "stablehlo.cholesky" in errortxt


def check_fft(_, mlirbc, __):
  return "stablehlo.fft" in mlirbc


def check_index_constant(errortxt, _, __):
  return re.search(
      "'vm.trunc.i64.i32' op operand #[0-9]* must be 64-bit signless integer, but got 'index'",
      errortxt)


def check_index_cast(errortxt, _, __):
  return re.search(
      "'arith.index_cast' op operand type 'i[0-9]*' and result type 'i[0-9]*' are cast incompatible",
      errortxt)


def check_schedule_allocation(errortxt, _, __):
  return "Pipeline failed while executing [`ScheduleAllocation`" in errortxt


def check_dot_i1(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    has_i1 = re.search("tensor<([0-9]*x)*i1>", line)
    has_dot = re.search("stablehlo.dot", line)
    if has_i1 and has_dot:
      return True
  return False


def check_vectorize(errortxt, _, __):
  return "arith.truncf' op operand #0 must be floating-point-like, but got 'vector<f32>" in errortxt


def check_roundeven(errortxt, _, __):
  return "roundeven" in errortxt


def check_numerical(errortxt, _, __):
  return "Mismatched elements" in errortxt


def check_compilation(errortxt, _, __):
  return "iree/integrations/pjrt/common/api_impl.cc:1085" in errortxt


def check_scatter(errortxt, _, __):
  return "'iree_linalg_ext.scatter' op mismatch in shape of indices and update value at dim#0" in errortxt


def check_shape_cast(errortxt, _, __):
  return "'vector.shape_cast' op source/result number of elements must match" in errortxt


def check_scatter_crash(_, mlirbc, runtime_crash):
  return "stablehlo.scatter" in mlirbc and runtime_crash


def check_eigen_decomposition(errortxt, _, __):
  return "Nonsymmetric eigendecomposition is only implemented on the CPU backend" in errortxt


def check_jax_unimplemented(errortxt, _, __):
  return "NotImplementedError: MLIR translation rule for primitive" in errortxt


def check_serialize_exe(errortxt, _, __):
  return "UNIMPLEMENTED; PJRT_Executable_Serialize" in errortxt


def check_optimized_prgrm(errortxt, _, __):
  return "UNIMPLEMENTED; PJRT_Executable_OptimizedProgram" in errortxt


def check_transpose_transfer(errortxt, _, __):
  return "UNIMPLEMENTED; only dense, row-major layouts currently supported" in errortxt


def check_optimized_program(errortxt, _, __):
  return "UNIMPLEMENTED; PJRT_Executable_OptimizedProgram" in errortxt


def check_donation(errortxt, _, __):
  return "Donation is not implemented for iree_cpu" in errortxt


def check_semaphore_overload(errortxt, _, __):
  return "OUT_OF_RANGE; semaphore values must be monotonically increasing;" in errortxt


def check_python_callback(errortxt, _, __):
  return "ValueError: `EmitPythonCallback` not supported" in errortxt


def check_complex_convolution(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    has_i1 = re.search("tensor<([0-9]*x)*complex<f[0-9]*>>", line)
    has_conv = re.search("stablehlo.convolution", line)
    has_dot = re.search("stablehlo.dot", line)
    if has_i1 and (has_conv or has_dot):
      return True
  return False


def check_unsigned_topk(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    has_topk = re.search("chlo.topk", line)
    unsigned = re.search(": tensor<[0-9]*xui[0-9]* ->", line)
    if has_topk and unsigned:
      return True
  return False


def check_unknown_backend(errortxt, _, __):
  return "RuntimeError: Unknown backend" in errortxt


def check_runtime_crash(__, _, runtime_crash):
  return runtime_crash


KnownChecks = {
    "https://github.com/openxla/iree/issues/????? (unknown)":
        check_jax_unimplemented,
    "https://github.com/openxla/iree/issues/12408 (transpose mem)":
        check_transpose_transfer,
    "https://github.com/openxla/iree/issues/13726 (collective)":
        check_collective,
    "https://github.com/openxla/iree/issues/12410 (custom call)":
        check_custom_call,
    "https://github.com/openxla/iree/issues/11018 (triangle)":
        check_triangular_solve,
    "https://github.com/openxla/iree/issues/14077 (index constant)":
        check_index_constant,
    "https://github.com/openxla/iree/issues/12263 (fft)":
        check_fft,
    "https://github.com/openxla/iree/issues/14072 (complex convolution)":
        check_complex_convolution,
    "https://github.com/openxla/iree/issues/14090 (index cast)":
        check_index_cast,
    "https://github.com/openxla/iree/issues/10816 (cholesky)":
        check_cholesky,
    "https://github.com/openxla/iree/issues/11761 (rng bit gen)":
        check_rng_bit,
    "https://github.com/openxla/iree/issues/????? (eigen decomp)":
        check_eigen_decomposition,
    "https://github.com/openxla/iree/issues/13579 (scatter ui)":
        check_scatter_ui,
    "https://github.com/openxla/iree/issues/13725 (cross repl)":
        check_cross_replica,
    "https://github.com/openxla/iree/issues/13727 (optbarrier)":
        check_optimization_barrier,
    "https://github.com/openxla/iree/issues/13493 (dot i1)":
        check_dot_i1,
    "https://github.com/openxla/iree/issues/13522 (roundeven)":
        check_roundeven,
    "https://github.com/openxla/iree/issues/13577 (max/min f16)":
        check_min_max_f16,
    "https://github.com/openxla/iree/issues/13523 (scatter)":
        check_scatter,
    "https://github.com/openxla/iree/issues/13580 (scatter crash)":
        check_scatter_crash,
    "https://github.com/openxla/iree/issues/14079 (shape_cast)":
        check_shape_cast,
    "https://github.com/openxla/iree/issues/????? (optimized prgrm)":
        check_optimized_program,
    "https://github.com/openxla/iree/issues/????? (donation)":
        check_donation,
    "https://github.com/openxla/iree/issues/????? (python callback)":
        check_python_callback,
    "https://github.com/openxla/iree/issues/14098 (unsigned topk)":
        check_unsigned_topk,
    "(unknown backend)":
        check_unknown_backend,
    "(semaphore)":
        check_semaphore_overload,
    "Runtime Crash":
        check_runtime_crash,
    "Compilation Failure":
        check_compilation,
    "Numerical Failures":
        check_numerical,
    "Untriaged":
        lambda _, __, ___: True,
}


def triage_test(test):
  files = sorted(os.listdir(f'{args.logdir}/{test}'))
  # Load the error.txt if it is available.
  error = ""
  if "error.txt" in files:
    with open(f'{args.logdir}/{test}/error.txt') as f:
      error = "".join(f.readlines())

  # Load the last bytecode file that was attempted to be compiled:
  mlirbc_count = len([f for f in files if "mlirbc" in f])
  mlirbc_name = f'{mlirbc_count - 1}-program.mlirbc'
  vmfb_name = f'{mlirbc_count - 1}-program.vmfb'

  runtime_crash = "CRASH_MARKER" in files and (mlirbc_count == 0 or
                                               vmfb_name in files)

  mlirbc = ""
  if mlirbc_count > 0:
    with mlir.make_ir_context() as ctx:
      with open(f'{args.logdir}/{test}/{mlirbc_name}', 'rb') as f:
        mlirbc = f.read()
      mlirbc = str(mlir_ir.Module.parse(mlirbc))

  for checkname in KnownChecks:
    if KnownChecks[checkname](error, mlirbc, runtime_crash):
      return checkname

  return "unknown error"


def filter_error_mapping(tests):
  error_mapping = {}
  with multiprocessing.Pool(int(args.jobs) if args.jobs else args.jobs) as p:
    results = p.map(triage_test, tests)

  for test, result in zip(tests, results):
    error_mapping[test] = result
  return error_mapping


def generate_summary(mapping):
  summary = {}
  for err in KnownChecks.keys():
    summary[err] = []
  for test in mapping:
    summary[mapping[test]].append(test)
  return summary


def print_summary(summary):
  maxlen = 0
  for error in summary:
    maxlen = max(len(error), maxlen)
  for error in summary:
    print(f'{error:<{maxlen}} : {len(summary[error])}')

  passstr = "Passing"
  failstr = "Failing"
  print(f'{passstr:<{maxlen}} : {len(tests) - len(failing)}')
  print(f'{failstr:<{maxlen}} : {len(failing)}')


failing = filter_to_failures(tests)
mapping = filter_error_mapping(failing)
summary = generate_summary(mapping)
print_summary(summary)

for test in summary["Untriaged"]:
  print(test)
