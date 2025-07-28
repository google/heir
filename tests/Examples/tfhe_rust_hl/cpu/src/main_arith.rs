#[cfg(test)]
mod test {
    use arith_test_rs_lib;

    use tfhe::prelude::*;
    use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint32};

    #[test]
    fn simple_test() {
        // Input = 27, output = 33 * 27 + 429 = 1320
        let config = ConfigBuilder::default().build();

        // Client-side
        let (client_key, server_key) = generate_keys(config);

        let a: tfhe::FheUint<tfhe::FheUint32Id> = FheUint32::encrypt(27u8, &client_key);
        let input_vec = core::array::from_fn(|_3| core::array::from_fn(|_2| a.clone()));

        set_server_key(server_key);

        let result = arith_test_rs_lib::fn_under_test(&input_vec);

        let output: u32 = result[0][0].decrypt(&client_key);
        assert_eq!(output, 1320);
    }
}
