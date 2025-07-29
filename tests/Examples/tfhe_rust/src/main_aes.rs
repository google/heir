#[allow(unused_imports)]
use std::time::Instant;
use tfhe::shortint::prelude::*;
use tfhe::shortint::*;

use add_round_key_rs_lib;
use mix_columns_rs_lib;
use shift_rows_rs_lib;
use sub_bytes_rs_lib;
use inv_mix_columns_rs_lib;
use inv_shift_rows_rs_lib;
use inv_sub_bytes_rs_lib;

type SecretBlock = [[Ciphertext; 8]; 16];

// Encrypt a list of 16 u8s
pub fn encrypt_u8(value: u8, client_key: &ClientKey) -> [Ciphertext; 8] {
    core::array::from_fn(|shift| {
        let bit = (value >> shift) & 1;
        client_key.encrypt(if bit != 0 { 1 } else { 0 })
    })
}

// Decrypt a list of 16 u8s
pub fn decrypt_u8(ciphertexts: &[Ciphertext; 8], client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        accum |= (bit as u8) << i;
    }
    accum
}

pub fn encrypt_block(value: [u8; 16], client_key: &ClientKey) -> [[Ciphertext; 8]; 16] {
    let c_vec: Vec<[Ciphertext; 8]> =
        value.into_iter().map(|v| encrypt_u8(v, client_key)).collect();
    let c_arr: [[Ciphertext; 8]; 16] = c_vec.try_into().expect("Failed to convert to array");
    c_arr
}

pub fn decrypt_block(ciphertexts: &[[Ciphertext; 8]; 16], client_key: &ClientKey) -> [u8; 16] {
    let p_vec: Vec<u8> = ciphertexts.into_iter().map(|ct| decrypt_u8(ct, client_key)).collect();
    let p_arr: [u8; 16] = p_vec.try_into().expect("Failed to convert to array");
    p_arr
}

pub fn aes_encrypt_block(
    server_key: &ServerKey,
    block: &SecretBlock,
    key: &[SecretBlock; 11],
    n_rounds: usize,
) -> SecretBlock {
    let mut block = add_round_key_rs_lib::add_round_key(&server_key, &block, &key[0]);
    for i in 1..(n_rounds + 1) {
        block = sub_bytes_rs_lib::sub_bytes(&server_key, &block);
        block = shift_rows_rs_lib::shift_rows(&block);
        block = mix_columns_rs_lib::mix_columns(&server_key, &block);
        block = add_round_key_rs_lib::add_round_key(&server_key, &block, &key[i]);
    }
    block = sub_bytes_rs_lib::sub_bytes(&server_key, &block);
    block = shift_rows_rs_lib::shift_rows(&block);
    block = mix_columns_rs_lib::mix_columns(&server_key, &block);
    block = add_round_key_rs_lib::add_round_key(&server_key, &block, &key[n_rounds + 1]);
    block
}

pub fn aes_decrypt_block(
    server_key: &ServerKey,
    block: &SecretBlock,
    key: &[SecretBlock; 11],
    n_rounds: usize,
) -> SecretBlock {
    let mut block = add_round_key_rs_lib::add_round_key(&server_key, &block, &key[n_rounds + 1]);
    block = inv_mix_columns_rs_lib::inv_mix_columns(&server_key, &block);
    block = inv_shift_rows_rs_lib::inv_shift_rows(&block);
    block = inv_sub_bytes_rs_lib::inv_sub_bytes(&server_key, &block);

    for i in 1..(n_rounds + 1) {
        block = add_round_key_rs_lib::add_round_key(&server_key, &block, &key[n_rounds+1-i]);
        block = inv_mix_columns_rs_lib::inv_mix_columns(&server_key, &block);
        block = inv_shift_rows_rs_lib::inv_shift_rows(&block);
        block = inv_sub_bytes_rs_lib::inv_sub_bytes(&server_key, &block);
    }
    block = add_round_key_rs_lib::add_round_key(&server_key, &block, &key[0]);
    block
}

fn main() {
    let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);

    let block = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    // expanded key client side
    let expanded_key = [
        [15, 21, 113, 201, 71, 217, 232, 89, 12, 183, 173, 214, 175, 127, 103, 152],
        [220, 144, 55, 176, 155, 73, 223, 233, 151, 254, 114, 63, 56, 129, 21, 167],
        [210, 201, 107, 183, 73, 128, 180, 94, 222, 126, 198, 97, 230, 255, 211, 198],
        [192, 175, 223, 57, 137, 47, 107, 103, 87, 81, 173, 6, 177, 174, 126, 192],
        [44, 92, 101, 241, 165, 115, 14, 150, 242, 34, 163, 144, 67, 140, 221, 80],
        [88, 157, 54, 235, 253, 238, 56, 125, 15, 204, 155, 237, 76, 64, 70, 189],
        [113, 199, 76, 194, 140, 41, 116, 191, 131, 229, 239, 82, 207, 165, 169, 239],
        [55, 20, 147, 72, 187, 61, 231, 247, 56, 216, 8, 165, 247, 125, 161, 74],
        [72, 38, 69, 32, 243, 27, 162, 215, 203, 195, 170, 114, 60, 190, 11, 56],
        [253, 13, 66, 203, 14, 22, 224, 28, 197, 213, 74, 110, 249, 107, 65, 86],
        [180, 142, 243, 82, 186, 152, 19, 78, 127, 77, 89, 32, 134, 38, 24, 118],
    ];

    let ct_block = encrypt_block(block, &client_key);
    let ct_vec_expanded_key: Vec<[[Ciphertext; 8]; 16]> =
        expanded_key.into_iter().map(|v| encrypt_block(v, &client_key)).collect();
    let ct_expaned_key: [[[Ciphertext; 8]; 16]; 11] =
        ct_vec_expanded_key.try_into().expect("Failed to convert to array");

    println!("input: {:?}", block);
    let mut t = Instant::now();
    let aes_encrypted = aes_encrypt_block(&server_key, &ct_block, &ct_expaned_key, 9);
    let mut run = t.elapsed().as_millis();
    println!("{:?} aes encryption time", run);
    t = Instant::now();
    let aes_decrypted = aes_decrypt_block(&server_key, &aes_encrypted, &ct_expaned_key, 9);
     run = t.elapsed().as_millis();
    println!("{:?} aes decryption time", run);
    let output = decrypt_block(&aes_decrypted, &client_key);
    println!("output: {:?}", output);
}
