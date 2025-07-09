#[cfg(test)]
mod test {
    use tfhe::shortint::prelude::*;

    use bitand_test_rs_lib;

    #[test]
    fn simple_test() {
        let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);

        // bit and 0b10 and 0b11
        let ct_1 = client_key.encrypt(2);
        let ct_2 = client_key.encrypt(3);

        let result = bitand_test_rs_lib::fn_under_test(&server_key, &ct_1, &ct_2);
        let output = client_key.decrypt(&result);
        assert_eq!(output, 2);
    }
}
