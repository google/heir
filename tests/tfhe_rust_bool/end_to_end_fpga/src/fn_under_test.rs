use tfhe::boolean::prelude::*;


// pub fn fn_under_test(
//   v0: &ServerKey,
//   v1: &Vec<Ciphertext>,
//   v2: &Vec<Ciphertext>,
// ) -> Vec<Ciphertext> {
//   let v1 = v1.iter().collect();
//   let v2 = v2.iter().collect();
//   let v3 = v0.xor_packed(&v1, &v2);
//   v3
// }

use tfhe::boolean::prelude::*;

pub fn fn_under_test(
  v0: &ServerKey,
  v1: &Vec<Ciphertext>,
  v2: &Vec<Ciphertext>,
) -> Vec<Ciphertext> {
  let v3 = 7;
  let v4 = 6;
  let v5 = 5;
  let v6 = 4;
  let v7 = 3;
  let v8 = 2;
  let v9 = 1;
  let v10 = 0;
  let v11 = vec![&v1[0]];
  let v12 = vec![&v1[1]];
  let v13 = vec![&v1[2]];
  let v14 = vec![&v1[3]];
  let v15 = vec![&v1[4]];
  let v16 = vec![&v1[5]];
  let v17 = vec![&v1[6]];
  let v18 = vec![&v1[7]];
  let v19 = vec![&v2[0]];
  let v20 = vec![&v2[1]];
  let v21 = vec![&v2[2]];
  let v22 = vec![&v2[3]];
  let v23 = vec![&v2[4]];
  let v24 = vec![&v2[5]];
  let v25 = vec![&v2[6]];
  let v26 = vec![&v2[7]];
  let v27_ref = v0.xor_packed(&v11, &v19);
  let v27: Vec<&Ciphertext> = v27_ref.iter().collect();
  let v28_ref = v0.and_packed(&v11, &v19);
  let v28: Vec<&Ciphertext> = v28_ref.iter().collect();
  let v29_ref = v0.xor_packed(&v12, &v20);
  let v29: Vec<&Ciphertext> = v29_ref.iter().collect();
  let v30_ref = v0.and_packed(&v12, &v20);
  let v30: Vec<&Ciphertext> = v30_ref.iter().collect();
  let v31_ref = v0.and_packed(&v29, &v28);
  let v31: Vec<&Ciphertext> = v31_ref.iter().collect();
  let v32_ref = v0.xor_packed(&v29, &v28);
  let v32: Vec<&Ciphertext> = v32_ref.iter().collect();
  let v33_ref = v0.xor_packed(&v30, &v31);
  let v33: Vec<&Ciphertext> = v33_ref.iter().collect();
  let v34_ref = v0.xor_packed(&v13, &v21);
  let v34: Vec<&Ciphertext> = v34_ref.iter().collect();
  let v35_ref = v0.and_packed(&v13, &v21);
  let v35: Vec<&Ciphertext> = v35_ref.iter().collect();
  let v36_ref = v0.and_packed(&v34, &v33);
  let v36: Vec<&Ciphertext> = v36_ref.iter().collect();
  let v37_ref = v0.xor_packed(&v34, &v33);
  let v37: Vec<&Ciphertext> = v37_ref.iter().collect();
  let v38_ref = v0.xor_packed(&v35, &v36);
  let v38: Vec<&Ciphertext> = v38_ref.iter().collect();
  let v39_ref = v0.xor_packed(&v14, &v22);
  let v39: Vec<&Ciphertext> = v39_ref.iter().collect();
  let v40_ref = v0.and_packed(&v14, &v22);
  let v40: Vec<&Ciphertext> = v40_ref.iter().collect();
  let v41_ref = v0.and_packed(&v39, &v38);
  let v41: Vec<&Ciphertext> = v41_ref.iter().collect();
  let v42_ref = v0.xor_packed(&v39, &v38);
  let v42: Vec<&Ciphertext> = v42_ref.iter().collect();
  let v43_ref = v0.xor_packed(&v40, &v41);
  let v43: Vec<&Ciphertext> = v43_ref.iter().collect();
  let v44_ref = v0.xor_packed(&v15, &v23);
  let v44: Vec<&Ciphertext> = v44_ref.iter().collect();
  let v45_ref = v0.and_packed(&v15, &v23);
  let v45: Vec<&Ciphertext> = v45_ref.iter().collect();
  let v46_ref = v0.and_packed(&v44, &v43);
  let v46: Vec<&Ciphertext> = v46_ref.iter().collect();
  let v47_ref = v0.xor_packed(&v44, &v43);
  let v47: Vec<&Ciphertext> = v47_ref.iter().collect();
  let v48_ref = v0.xor_packed(&v45, &v46);
  let v48: Vec<&Ciphertext> = v48_ref.iter().collect();
  let v49_ref = v0.xor_packed(&v16, &v24);
  let v49: Vec<&Ciphertext> = v49_ref.iter().collect();
  let v50_ref = v0.and_packed(&v16, &v24);
  let v50: Vec<&Ciphertext> = v50_ref.iter().collect();
  let v51_ref = v0.and_packed(&v49, &v48);
  let v51: Vec<&Ciphertext> = v51_ref.iter().collect();
  let v52_ref = v0.xor_packed(&v49, &v48);
  let v52: Vec<&Ciphertext> = v52_ref.iter().collect();
  let v53_ref = v0.xor_packed(&v50, &v51);
  let v53: Vec<&Ciphertext> = v53_ref.iter().collect();
  let v54_ref = v0.xor_packed(&v17, &v25);
  let v54: Vec<&Ciphertext> = v54_ref.iter().collect();
  let v55_ref = v0.and_packed(&v17, &v25);
  let v55: Vec<&Ciphertext> = v55_ref.iter().collect();
  let v56_ref = v0.and_packed(&v54, &v53);
  let v56: Vec<&Ciphertext> = v56_ref.iter().collect();
  let v57_ref = v0.xor_packed(&v54, &v53);
  let v57: Vec<&Ciphertext> = v57_ref.iter().collect();
  let v58_ref = v0.xor_packed(&v55, &v56);
  let v58: Vec<&Ciphertext> = v58_ref.iter().collect();
  let v59_ref = v0.xor_packed(&v18, &v26);
  let v59: Vec<&Ciphertext> = v59_ref.iter().collect();
  let v60_ref = v0.xor_packed(&v59, &v58);
  let v60: Vec<&Ciphertext> = v60_ref.iter().collect();
  let mut v61: Vec<Ciphertext> = vec![];
  v61.extend(v60_ref);
  v61.extend(v57_ref);
  v61.extend(v52_ref);
  v61.extend(v47_ref);
  v61.extend(v42_ref);
  v61.extend(v37_ref);
  v61.extend(v32_ref);
  v61.extend(v27_ref);
  v61
}