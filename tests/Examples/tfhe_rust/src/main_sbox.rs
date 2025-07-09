#[allow(unused_imports)]
use tfhe::shortint::*;

pub fn encrypt_u8(value: u8, client_key: &ClientKey) -> [Ciphertext; 8] {
    core::array::from_fn(|shift| {
        let bit = (value >> shift) & 1;
        client_key.encrypt(if bit != 0 { 1 } else { 0 })
    })
}

pub fn decrypt_u8(ciphertexts: &[Ciphertext; 8], client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        accum |= (bit as u8) << i;
    }
    accum
}

#[cfg(test)]
mod test {
    use tfhe::shortint::parameters::v1_1::V1_1_PARAM_MESSAGE_3_CARRY_1_KS_PBS_GAUSSIAN_2M128;
    use tfhe::shortint::prelude::*;
    use tfhe::shortint::*;

    use super::decrypt_u8;
    use super::encrypt_u8;

    use sbox_test_rs_lib;

    #[test]
    fn simple_test() {
        let (client_key, server_key) = gen_keys(V1_1_PARAM_MESSAGE_3_CARRY_1_KS_PBS_GAUSSIAN_2M128);

        // query the first 2 and last 2 elements of the table
        let query = [0, 1, 6, 7];
        let expected = [99, 124, 111, 197];
        let ct_query: Vec<[Ciphertext; 8]> =
            query.into_iter().map(|v| encrypt_u8(v, &client_key)).collect();

        for i in 0..4 {
            let result = sbox_test_rs_lib::sub_bytes(&server_key, &ct_query[i]);
            let output = decrypt_u8(&result, &client_key);
            assert_eq!(output, expected[i]);
        }
    }
}
