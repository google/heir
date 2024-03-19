use tfhe::boolean::prelude::*;


pub fn fn_under_test(
  v0: &ServerKey,
  v1: &Vec<Ciphertext>,
  v2: &Vec<Ciphertext>,
) -> Vec<Ciphertext> {
  let v1 = v1.iter().collect();
  let v2 = v2.iter().collect();
  let v3 = v0.xor_packed(&v1, &v2);
  v3
}


// pub fn fn_under_test(
//   v0: &ServerKey,
//   v1: &Vec<&Ciphertext>,
//   v2: &Vec<&Ciphertext>,
// ) -> Vec<Ciphertext> {
//   let v3 = 7;
//   let v4 = 6;
//   let v5 = 5;
//   let v6 = 4;
//   let v7 = 3;
//   let v8 = 2;
//   let v9 = 1;
//   let v10 = 0;
//   let v11 = vec![v1[v10]];
//   let v12 = vec![v1[v9]];
//   let v13 = vec![v1[v8]];
//   let v14 = vec![v1[v7]];
//   let v15 = vec![v1[v6]];
//   let v16 = vec![v1[v5]];
//   let v17 = vec![v1[v4]];
//   let v18 = vec![v1[v3]];
//   let v19 = vec![v2[v10]];
//   let v20 = vec![v2[v9]];
//   let v21 = vec![v2[v8]];
//   let v22 = vec![v2[v7]];
//   let v23 = vec![v2[v6]];
//   let v24 = vec![v2[v5]];
//   let v25 = vec![v2[v4]];
//   let v26 = vec![v2[v3]];
//   let v27 = v0.xor_packed(&v11, &v19);
//   let v27 = v27.iter().collect();
//   let v28 = v0.and_packed(&v11, &v19);
//   let v29 = v0.xor_packed(&v12, &v20);
//   let v30 = v0.and_packed(&v12, &v20);
//   let v28 = v28.iter().collect();
//   let v29  = v29.iter().collect();
//   let v31 = v0.and_packed(&v29, &v28);
//   let v32 = v0.xor_packed(&v29, &v28);
//   let v33 = v0.xor_packed(&v30, &v31);
//   let v34 = v0.xor_packed(&v13, &v21);
//   let v35 = v0.and_packed(&v13, &v21);
//   let v36 = v0.and_packed(&v34, &v33);
//   let v37 = v0.xor_packed(&v34, &v33);
//   let v38 = v0.xor_packed(&v35, &v36);
//   let v39 = v0.xor_packed(&v14, &v22);
//   let v40 = v0.and_packed(&v14, &v22);
//   let v41 = v0.and_packed(&v39, &v38);
//   let v42 = v0.xor_packed(&v39, &v38);
//   let v43 = v0.xor_packed(&v40, &v41);
//   let v44 = v0.xor_packed(&v15, &v23);
//   let v45 = v0.and_packed(&v15, &v23);
//   let v46 = v0.and_packed(&v44, &v43);
//   let v47 = v0.xor_packed(&v44, &v43);
//   let v48 = v0.xor_packed(&v45, &v46);
//   let v49 = v0.xor_packed(&v16, &v24);
//   let v50 = v0.and_packed(&v16, &v24);
//   let v51 = v0.and_packed(&v49, &v48);
//   let v52 = v0.xor_packed(&v49, &v48);
//   let v53 = v0.xor_packed(&v50, &v51);
//   let v54 = v0.xor_packed(&v17, &v25);
//   let v55 = v0.and_packed(&v17, &v25);
//   let v56 = v0.and_packed(&v54, &v53);
//   let v57 = v0.xor_packed(&v54, &v53);
//   let v58 = v0.xor_packed(&v55, &v56);
//   let v59 = v0.xor_packed(&v18, &v26);
//   let v60 = v0.xor_packed(&v59, &v58);
//   let v61 = vec![v60[0], v57[0], v52[0], v47[0], v42[0], v37[0], v32[0], v27[0]];
//   v61
// }
