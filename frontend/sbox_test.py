"""Example of HEIR Python usage."""

from heir import compile
from heir.mlir import F32, I1, I8, I16, I64, Secret, Tensor


# This example does not run end to end, but is used to generate MLIR for the
# CGGI pipeline
@compile(debug=True, scheme="cggi")
def func(x: Secret[I8]):
  x0: I1 = (x >> 7) & 1
  x1 = (x >> 6) & 1
  x2 = (x >> 5) & 1
  x3 = (x >> 4) & 1
  x4 = (x >> 3) & 1
  x5 = (x >> 2) & 1
  x6 = (x >> 1) & 1
  x7 = x & 1

  y14 = x3 ^ x5
  y13 = x0 ^ x6
  y9 = x0 ^ x3
  y8 = x0 ^ x5
  t0 = x1 ^ x2
  y1 = t0 ^ x7
  y4 = y1 ^ x3
  y12 = y13 ^ y14
  y2 = y1 ^ x0
  y5 = y1 ^ x6
  y3 = y5 ^ y8
  t1 = x4 ^ y12
  y15 = t1 ^ x5
  y20 = t1 ^ x1
  y6 = y15 ^ x7
  y10 = y15 ^ t0
  y11 = y20 ^ y9
  y7 = x7 ^ y11
  y17 = y10 ^ y11
  y19 = y10 ^ y8
  y16 = t0 ^ y11
  y21 = y13 ^ y16
  y18 = x0 ^ y16
  # nonlinear
  t2 = y12 & y15
  t3 = y3 & y6
  t4 = t3 ^ t2
  t5 = y4 & x7
  t6 = t5 ^ t2
  t7 = y13 & y16
  t8 = y5 & y1
  t9 = t8 ^ t7
  t10 = y2 & y7
  t11 = t10 ^ t7
  t12 = y9 & y11
  t13 = y14 & y17
  t14 = t13 ^ t12
  t15 = y8 & y10
  t16 = t15 ^ t12
  t17 = t4 ^ t14
  t18 = t6 ^ t16
  t19 = t9 ^ t14
  t20 = t11 ^ t16
  t21 = t17 ^ y20
  t22 = t18 ^ y19
  t23 = t19 ^ y21
  t24 = t20 ^ y18

  t25 = t21 ^ t22
  t26 = t21 & t23
  t27 = t24 ^ t26
  t28 = t25 & t27
  t29 = t28 ^ t22
  t30 = t23 ^ t24
  t31 = t22 ^ t26
  t32 = t31 & t30
  t33 = t32 ^ t24
  t34 = t23 ^ t33
  t35 = t27 ^ t33
  t36 = t24 & t35
  t37 = t36 ^ t34
  t38 = t27 ^ t36
  t39 = t29 & t38
  t40 = t25 ^ t39

  t41 = t40 ^ t37
  t42 = t29 ^ t33
  t43 = t29 ^ t40
  t44 = t33 ^ t37
  t45 = t42 ^ t41
  z0 = t44 & y15
  z1 = t37 & y6
  z2 = t33 & x7
  z3 = t43 & y16
  z4 = t40 & y1
  z5 = t29 & y7
  z6 = t42 & y11
  z7 = t45 & y17
  z8 = t41 & y10
  z9 = t44 & y12
  z10 = t37 & y3
  z11 = t33 & y4
  z12 = t43 & y13
  z13 = t40 & y5
  z14 = t29 & y2
  z15 = t42 & y9
  z16 = t45 & y14
  z17 = t41 & y8

  t46 = z15 ^ z16
  t47 = z10 ^ z11
  t48 = z5 ^ z13
  t49 = z9 ^ z10
  t50 = z2 ^ z12
  t51 = z2 ^ z5
  t52 = z7 ^ z8
  t53 = z0 ^ z3
  t54 = z6 ^ z7
  t55 = z16 ^ z17
  t56 = z12 ^ t48
  t57 = t50 ^ t53
  t58 = z4 ^ t46
  t59 = z3 ^ t54
  t60 = t46 ^ t57
  t61 = z14 ^ t57
  t62 = t52 ^ t58
  t63 = t49 ^ t58
  t64 = z4 ^ t59
  t65 = t61 ^ t62
  t66 = z1 ^ t63
  s0 = t59 ^ t63
  s6 = t56 ^ ~t62
  s7 = t48 ^ ~t60
  t67 = t64 ^ t65
  s3 = t53 ^ t66
  s4 = t51 ^ t66
  s5 = t47 ^ t65
  s1 = t64 ^ ~s3
  s2 = t55 ^ ~t67

  q = s0
  q = q << 1
  q = q + s1
  q = q << 1
  q = q + s2
  q = q << 1
  q = q + s3
  q = q << 1
  q = q + s4
  q = q << 1
  q = q + s5
  q = q << 1
  q = q + s6
  q = q << 1
  q = q + s7

  return q
