use tfhe::boolean::engine::fpga::{BelfortBooleanServerKey, Gate};
use tfhe::boolean::prelude::*;

pub enum ServerKeyEnum {
    TypeSW(ServerKey),
    TypeFPGA(BelfortBooleanServerKey),
}

pub trait ServerKeyTrait {
    fn packed_gates<const N: usize>(
        &self,
        gates: &Vec<Gate>,
        cts_left: &[&Ciphertext; N],
        cts_right: &[&Ciphertext; N],
    ) -> [Ciphertext; N];
    fn not(&self, ct: &Ciphertext) -> Ciphertext;
    fn packed_not<const N: usize>(&self, cts: &[&Ciphertext; N]) -> [Ciphertext; N];
    fn trivial_encrypt(&self, value: bool) -> Ciphertext;
}

impl ServerKeyTrait for ServerKey {
    fn packed_gates<const N: usize>(
        &self,
        gates: &Vec<Gate>,
        cts_left: &[&Ciphertext; N],
        cts_right: &[&Ciphertext; N],
    ) -> [Ciphertext; N] {
        let result_vec = self.packed_gates(gates, &cts_left.to_vec(), &cts_right.to_vec());

        result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
            panic!("Expected {} elements, found {}", N, v.len())
        })
    }

    fn not(&self, ct: &Ciphertext) -> Ciphertext {
        return self.not(ct);
    }

    fn packed_not<const N: usize>(&self, cts: &[&Ciphertext; N]) -> [Ciphertext; N] {
        let result_vec = self.packed_not(&cts.to_vec());

        result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
            panic!("Expected {} elements, found {}", N, v.len())
        })
    }

    fn trivial_encrypt(&self, value: bool) -> Ciphertext {
        return self.trivial_encrypt(value);
    }
}

impl ServerKeyTrait for BelfortBooleanServerKey {
    fn packed_gates<const N: usize>(
        &self,
        gates: &Vec<Gate>,
        cts_left: &[&Ciphertext; N],
        cts_right: &[&Ciphertext; N],
    ) -> [Ciphertext; N] {
        let result_vec = self.packed_gates(gates, &cts_left.to_vec(), &cts_right.to_vec());

        result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
            panic!("Expected {} elements, found {}", N, v.len())
        })
    }

    fn not(&self, ct: &Ciphertext) -> Ciphertext {
        return self.not(ct);
    }

    fn packed_not<const N: usize>(&self, cts: &[&Ciphertext; N]) -> [Ciphertext; N] {
        let result_vec = self.packed_not(&cts.to_vec());

        result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
            panic!("Expected {} elements, found {}", N, v.len())
        })
    }

    fn trivial_encrypt(&self, value: bool) -> Ciphertext {
        return self.trivial_encrypt(value);
    }
}

impl ServerKeyTrait for ServerKeyEnum {
    fn packed_gates<const N: usize>(
        &self,
        gates: &Vec<Gate>,
        cts_left: &[&Ciphertext; N],
        cts_right: &[&Ciphertext; N],
    ) -> [Ciphertext; N] {
        match self {
            ServerKeyEnum::TypeSW(sk) => {
                let result_vec = sk.packed_gates(gates, &cts_left.to_vec(), &cts_right.to_vec());

                result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
                    panic!("Expected {} elements, found {}", N, v.len())
                })
            }
            ServerKeyEnum::TypeFPGA(sk) => {
                let result_vec = sk.packed_gates(gates, &cts_left.to_vec(), &cts_right.to_vec());

                result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
                    panic!("Expected {} elements, found {}", N, v.len())
                })
            }
        }
    }

    fn not(&self, ct: &Ciphertext) -> Ciphertext {
        match self {
            ServerKeyEnum::TypeSW(sk) => sk.not(ct),
            ServerKeyEnum::TypeFPGA(sk) => sk.not(ct),
        }
    }

    fn packed_not<const N: usize>(&self, cts: &[&Ciphertext; N]) -> [Ciphertext; N] {
        match self {
            ServerKeyEnum::TypeSW(sk) => {
                let result_vec = sk.packed_not(&cts.to_vec());

                result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
                    panic!("Expected {} elements, found {}", N, v.len())
                })
            }

            ServerKeyEnum::TypeFPGA(sk) => {
                let result_vec = sk.packed_not(&cts.to_vec());

                result_vec.try_into().unwrap_or_else(|v: Vec<Ciphertext>| {
                    panic!("Expected {} elements, found {}", N, v.len())
                })
            }
        }
    }

    fn trivial_encrypt(&self, value: bool) -> Ciphertext {
        match self {
            ServerKeyEnum::TypeSW(sk) => sk.trivial_encrypt(value),
            ServerKeyEnum::TypeFPGA(sk) => sk.trivial_encrypt(value),
        }
    }
}
