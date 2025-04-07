func.func @cmux(%a: i64, %b: i64, %cond: i1 {secret.secret}) -> (i64) {
    %2 = scf.if %cond -> (i64) {
      scf.yield %a : i64
    } else {
      scf.yield %b : i64
    }
    func.return %2 : i64
}
