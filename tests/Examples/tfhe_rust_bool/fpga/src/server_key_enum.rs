use tfhe::boolean::engine::fpga::{BelfortBooleanServerKey, Gate};
use tfhe::boolean::prelude::*;

pub enum ServerKeyEnum {
    TypeSW(ServerKey),
    TypeFPGA(BelfortBooleanServerKey),
}

pub trait ServerKeyTrait {
    fn packed_gates(&self, gates: &Vec<Gate>, cts_left: &Vec<&Ciphertext>, cts_right: &Vec<&Ciphertext>) -> Vec<Ciphertext>;
    fn not(&self, ct: &Ciphertext) -> Ciphertext;
    fn packed_not(&self, cts: &Vec<&Ciphertext>) -> Vec<Ciphertext>;
    fn trivial_encrypt(&self, value: bool) -> Ciphertext;
}


impl ServerKeyTrait for ServerKey {
    fn packed_gates(&self, gates: &Vec<Gate>, cts_left: &Vec<&Ciphertext>, cts_right: &Vec<&Ciphertext>) -> Vec<Ciphertext> {
        return self.packed_gates(gates, cts_left, cts_right);
    }

    fn not(&self, ct: &Ciphertext) -> Ciphertext {
        return self.not(ct);
    }

    fn packed_not(&self, cts: &Vec<&Ciphertext>) -> Vec<Ciphertext> {
        return self.packed_not(cts);
    }

    fn trivial_encrypt(&self, value: bool) -> Ciphertext {
        return self.trivial_encrypt(value);
    }
}

impl ServerKeyTrait for BelfortBooleanServerKey {
    fn packed_gates(&self, gates: &Vec<Gate>, cts_left: &Vec<&Ciphertext>, cts_right: &Vec<&Ciphertext>) -> Vec<Ciphertext> {
        return self.packed_gates(gates, cts_left, cts_right);
    }

    fn not(&self, ct: &Ciphertext) -> Ciphertext {
        return self.not(ct);
    }

    fn packed_not(&self, cts: &Vec<&Ciphertext>) -> Vec<Ciphertext> {
        return self.packed_not(cts);
    }

    fn trivial_encrypt(&self, value: bool) -> Ciphertext {
        return self.trivial_encrypt(value);
    }

}

impl ServerKeyTrait for ServerKeyEnum {
    fn packed_gates(&self, gates: &Vec<Gate>, cts_left: &Vec<&Ciphertext>, cts_right: &Vec<&Ciphertext>) -> Vec<Ciphertext> {
        match self {
            ServerKeyEnum::TypeSW(sk) => sk.packed_gates(gates, cts_left, cts_right),
            ServerKeyEnum::TypeFPGA(sk) => sk.packed_gates(gates, cts_left, cts_right),
        }
    }

    fn not(&self, ct: &Ciphertext) -> Ciphertext {
        match self {
            ServerKeyEnum::TypeSW(sk) => sk.not(ct),
            ServerKeyEnum::TypeFPGA(sk) => sk.not(ct),
        }
    }

    fn packed_not(&self, cts: &Vec<&Ciphertext>) -> Vec<Ciphertext> {
        match self {
            ServerKeyEnum::TypeSW(sk) => sk.packed_not(cts),
            ServerKeyEnum::TypeFPGA(sk) => sk.packed_not(cts),
        }
    }

    fn trivial_encrypt(&self, value: bool) -> Ciphertext {
        match self {
            ServerKeyEnum::TypeSW(sk) => sk.trivial_encrypt(value),
            ServerKeyEnum::TypeFPGA(sk) => sk.trivial_encrypt(value),
        }
    }
}
