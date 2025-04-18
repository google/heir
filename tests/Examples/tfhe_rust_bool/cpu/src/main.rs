#[cfg(test)]
mod test {
    use and_test_rs_lib;

    #[test]
    fn simple_test() {
        let (client_key, server_key) = tfhe::boolean::gen_keys();

        let ct_1 = client_key.encrypt(1 == 1u8);
        let ct_2 = client_key.encrypt(1 == 1u8);

        let result = and_test_rs_lib::fn_under_test(&server_key, &ct_1, &ct_2);
        let output: bool = client_key.decrypt(&result);

        assert_eq!(output, true);
    }
}
