#[cfg(test)]
mod test {
    use tfhe::shortint::parameters::v1_1::V1_1_PARAM_MESSAGE_3_CARRY_1_KS_PBS_GAUSSIAN_2M128;
    use tfhe::shortint::prelude::*;

    use add_test_rs_lib;

    #[test]
    fn simple_test() {
        let (client_key, server_key) = gen_keys(V1_1_PARAM_MESSAGE_3_CARRY_1_KS_PBS_GAUSSIAN_2M128);

        let ct_1 = client_key.encrypt(2);
        let ct_2 = client_key.encrypt(3);

        let result = add_test_rs_lib::fn_under_test(&server_key, &ct_1, &ct_2);
        let output = client_key.decrypt(&result);
        assert_eq!(output, 5);
    }
}
