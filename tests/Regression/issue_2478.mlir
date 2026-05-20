// RUN: heir-opt --secret-insert-mgmt-ckks=slot-number=1024 %s | FileCheck %s

// CHECK-NOT: Setting single value on multi-result op with no existing " "array attr!op->setAttr(attrName, attr);"' failed
"builtin.module"()({
  "func.func"()<{function_type=(!secret.secret<index>,!secret.secret<i16>)->(!secret.secret<i16>,!secret.secret<i16>),sym_name="f"}>({
  ^bb0(%a0:!secret.secret<index>,%a1:!secret.secret<i16>):
    %0="arith.constant"()<{value=0:index}>:()->index
    %1="arith.constant"()<{value=1:index}>:()->index
    %2="arith.constant"()<{value=0:i16}>:()->i16
    %3:2="secret.generic"(%a0,%a1)({
    ^bb0(%b0:index,%b1:i16):
      %4:2="scf.for"(%0,%b0,%1,%2,%2)({
      ^bb0(%c0:index,%c1:i16,%c2:i16):
        %5="arith.cmpi"(%c1,%b1)<{predicate=2:i64}>:(i16,i16)->i1
        %6:2="scf.if"(%5)({"scf.yield"(%c1,%c1):(i16,i16)->()},{"scf.yield"(%c2,%c2):(i16,i16)->()}):(i1)->(i16,i16)
        "scf.yield"(%6#0,%6#1):(i16,i16)->()
      }):(index,index,index,i16,i16)->(i16,i16)
      "secret.yield"(%4#0,%4#1):(i16,i16)->()
    }):(!secret.secret<index>,!secret.secret<i16>)->(!secret.secret<i16>,!secret.secret<i16>)
    "func.return"(%3#0,%3#1):(!secret.secret<i16>,!secret.secret<i16>)->()
  }):()->()
}):()->()
