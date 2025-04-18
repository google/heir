#[cfg(test)]
mod test {
    use fully_connected_test_rs_lib;

    use tfhe::{ConfigBuilder, generate_keys, set_server_key, FheUint8};
    use tfhe::prelude::*;

    #[test]
    fn simple_test() {
        let config = ConfigBuilder::default().build();

        // Client-side
        let (client_key, server_key) = generate_keys(config);

        let a: tfhe::FheUint<tfhe::FheUint8Id> = FheUint8::encrypt(27u8, &client_key);
        let input_vec = core::array::from_fn(|_1| core::array::from_fn(|_1| a.clone()));

        set_server_key(server_key);

        // let t = Instant::now();
        let result = fully_connected_test_rs_lib::fn_under_test(&input_vec);
        // let elapsed = t.elapsed();
        // println!("Time elapsed: {:?}", elapsed.as_secs_f32());

        let output: u32 = result[0][0].decrypt(&client_key);
        assert_eq!(output, 2 * 27 + 1);
    }
}
