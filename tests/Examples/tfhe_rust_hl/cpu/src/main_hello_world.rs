#[cfg(test)]
mod test {
    use hello_world_clean_xsmall_test_rs_lib;

    use tfhe::prelude::*;
    use tfhe::{generate_keys, set_server_key, ConfigBuilder, FheUint8};

    #[test]
    fn simple_test() {
        let config = ConfigBuilder::default().build();

        // Client-side
        let (client_key, server_key) = generate_keys(config);

        let input: u8 = 31;

        let a: tfhe::FheUint<tfhe::FheUint8Id> = FheUint8::encrypt(input, &client_key);
        let input_vec = core::array::from_fn(|_1| core::array::from_fn(|_1| a.clone()));

        set_server_key(server_key);

        // let t = Instant::now();
        let result = hello_world_clean_xsmall_test_rs_lib::fn_under_test(&input_vec);
        // let elapsed = t.elapsed();
        // println!("Time elapsed: {:?}", elapsed.as_secs_f32());

        let output: u16 = result[0][0].decrypt(&client_key);
        assert_eq!(output, input as u16 * 9 + 1);

        let output: u16 = result[0][1].decrypt(&client_key);
        assert_eq!(output, input as u16 * 54 + 2);

        let output: u16 = result[0][2].decrypt(&client_key);
        assert_eq!(output, input as u16 * 57 + 5438);
    }
}
