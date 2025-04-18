#[cfg(test)]
mod test {
    use tfhe::shortint::parameters::get_parameters_from_message_and_carry;
    use tfhe::shortint::*;

    use add_test_rs_lib;

    #[test]
    fn simple_test() {
        let parameters = get_parameters_from_message_and_carry((1 << 3) - 1, 2);
        let (client_key, server_key) = tfhe::shortint::gen_keys(parameters);

        let ct_1 = client_key.encrypt(2);
        let ct_2 = client_key.encrypt(3);

        let result = add_test_rs_lib::fn_under_test(&server_key, &ct_1, &ct_2);
        let output = client_key.decrypt(&result);
        assert_eq!(output, 5);

    }
}
