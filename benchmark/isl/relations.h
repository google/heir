#ifndef BENCHMARK_ISL_RELATIONS_H_
#define BENCHMARK_ISL_RELATIONS_H_

namespace mlir {
namespace heir {

// from Tricyclic encoding
inline constexpr char kRelation1[] =
    "{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (357i0 + 84i1 + 272i2 + "
    "slot) mod 714 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 16 and 0 <= i2 <= "
    "20 and 0 <= slot <= 8191 }";

inline constexpr char kRelation2[] =
    "{ [i0, i1, i2] -> [ct, slot] : (2048 + 98i0 - 100i1 - i2 + ct + "
    "2048*floor((-98 - 98i0 + slot)/2048)) mod 4096 = 0 and 0 <= i0 <= 15 "
    "and 0 <= i1 <= 21 and 0 <= i2 <= 1 and 0 <= ct <= 2047 and 0 <= slot "
    "<= 2146 and 2048*floor((-98 - 98i0 + slot)/2048) >= -3615 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) >= -2147 - 98i0 + i2 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + i2 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 + slot }";

inline constexpr char kRelation3[] =
    "{ [d0, d1, d2] -> [r0, r1] : exists (l0, l1 : -13*d0 + 21*d1 + d2 - "
    "r0 + 1024*l1 = 0 and 0 <= d0 <= 47 and 0 <= d1 <= 47 and 0 <= d2 <= "
    "8 and 0 <= r0 <= 1023 and 0 <= r1 <= 4095 and 13*d0 - r1 + 1024*l0 + "
    "1036 >= 0 and -13*d0 + r1 - 1024*l0 - 1024 >= 0 and r1 - 1024*l0 - "
    "1024 >= 0) }";

// --- Hotword layouts ---

// Filter layout for Conv 1 (1D Conv, 16 out, 40 in, kW 3)
inline constexpr char kLayout4Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : (2048 + 98i0 - 100i1 - i2 + ct + "
    "2048*floor((-98 - 98i0 + slot)/2048)) mod 4096 = 0 and 0 <= i0 <= 15 "
    "and 0 <= i1 <= 39 and 0 <= i2 <= 2 and 0 <= ct <= 2047 and 0 <= slot "
    "<= 4095 and 2048*floor((-98 - 98i0 + slot)/2048) >= -3615 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) >= -2147 - 98i0 + i2 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 - 98i0 + i2 + slot and "
    "2048*floor((-98 - 98i0 + slot)/2048) <= -2048 + slot }";

// Filter layout for Conv 2 (1D Conv, 24 out, 16 in, kW 1, stride 2)
inline constexpr char kLayout8Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e0, e1, e2, e4: i2 = 0 and "
    "2048e4 = 98i0 - 98i1 + ct + slot - 2e2 and 0 <= i0 <= 23 and 0 <= i1 "
    "<= 15 and 0 <= ct <= 2047 and 0 <= slot <= 4095 and e1 <= 11 and e2 "
    ">= 49i0 and 0 <= e2 <= 1175 and e2 <= 48 + 49i0 and "
    "2048*floor((872 + slot)/2048) >= slot - 2e0 + 2e1 and slot - 2e0 <= "
    "2048*floor((872 + slot)/2048) <= 1 + slot - 2e0 + 2e1 and "
    "2048*floor((872 + slot)/2048) <= slot and -48 + 49slot - 98e0 + e2 <= "
    "100352*floor((872 + slot)/2048) <= 49slot - 98e0 + e2 and 97slot - "
    "196e0 + 98e1 + 2e2 <= 198656*floor((872 + slot)/2048) <= 1 + 97slot "
    "- 196e0 + 98e1 + 2e2) }";

// Output layout for Conv 2 (1D Conv, shape 1x24x49)
inline constexpr char kLayout10Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e1, e2: i0 = 0 and ct = 0 and "
    "0 <= i1 <= 23 and 0 <= i2 <= 48 and 0 <= slot <= 4095 and "
    "2048*floor((-1 - 49i1 - i2)/2048) <= -873 - 49i1 - i2 and e2 <= 11 "
    "and 2048*floor((872 + slot)/2048) >= slot - 2e1 + 2e2 and slot - 2e1 "
    "<= 2048*floor((872 + slot)/2048) <= 1 + slot - 2e1 + 2e2 and "
    "2048*floor((872 + slot)/2048) <= slot and 2000 + 49i1 + i2 + 49slot "
    "+ 2048*floor((-1 - 49i1 - i2)/2048) - 98e1 <= "
    "100352*floor((872 + slot)/2048) <= 2048 + 49i1 + i2 + 49slot + "
    "2048*floor((-1 - 49i1 - i2)/2048) - 98e1 and 4096 + 98i1 + 2i2 + "
    "97slot + 4096*floor((-1 - 49i1 - i2)/2048) - 196e1 + 98e2 <= "
    "198656*floor((872 + slot)/2048) <= 4097 + 98i1 + 2i2 + 97slot + "
    "4096*floor((-1 - 49i1 - i2)/2048) - 196e1 + 98e2) }";

// Filter layout for Conv 3 (1D Conv, 24 out, 16 in, kW 9, stride 2)
inline constexpr char kLayout13Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e0, e1, e2, e4: 2048e4 = 98i0 "
    "- 106i1 - i2 + ct + slot - 2e2 and 0 <= i0 <= 23 and 0 <= i1 <= 15 "
    "and 0 <= i2 <= 8 and 0 <= ct <= 2047 and 0 <= slot <= 4095 and e1 <= "
    "11 and e2 >= 49i0 and 0 <= e2 <= 1175 and e2 <= 48 + 49i0 and 98i0 - "
    "i2 <= 2e2 <= 105 + 98i0 - i2 and 2048*floor((872 + slot)/2048) >= "
    "slot - 2e0 + 2e1 and slot - 2e0 <= 2048*floor((872 + slot)/2048) <= "
    "1 + slot - 2e0 + 2e1 and 2048*floor((872 + slot)/2048) <= slot and "
    "-48 + 49slot - 98e0 + e2 <= 100352*floor((872 + slot)/2048) <= 49slot "
    "- 98e0 + e2 and 97slot - 196e0 + 98e1 + 2e2 <= "
    "198656*floor((872 + slot)/2048) <= 1 + 97slot - 196e0 + 98e1 + 2e2) }";

// Filter layout for Conv 4 (1D Conv, 24 out, 24 in, kW 9)
inline constexpr char kLayout17Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : (49i0 - 57i1 - i2 + ct) mod 2048 = 0 "
    "and 0 <= i0 <= 23 and 0 <= i1 <= 23 and 0 <= i2 <= 8 and 0 <= ct <= "
    "2047 and 0 <= slot <= 4095 and 2048*floor((-49 - 49i0 + slot)/2048) "
    ">= -3223 + slot and 2048*floor((-49 - 49i0 + slot)/2048) >= -2104 - "
    "49i0 + i2 + slot and 2048*floor((-49 - 49i0 + slot)/2048) <= -2048 - "
    "49i0 + slot and 2048*floor((-49 - 49i0 + slot)/2048) <= -2048 - 49i0 "
    "+ i2 + slot and 2048*floor((-49 - 49i0 + slot)/2048) <= -2048 + slot }";

// Filter layout for Conv 5 (1D Conv, 32 out, 24 in, kW 1, stride 2)
inline constexpr char kLayout19Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e0, e1, e2, e4: i2 = 0 and "
    "2048e4 = 50i0 - 49i1 + ct + slot - 2e2 and 0 <= i0 <= 31 and 0 <= i1 "
    "<= 23 and 0 <= ct <= 1023 and 0 <= slot <= 4095 and e1 <= 15 and e2 "
    ">= 25i0 and 0 <= e2 <= 799 and e2 <= 24 + 25i0 and 50i0 - 49i1 <= 2e2 "
    "<= 1175 + 50i0 - 49i1 and 1024*floor((224 + slot)/1024) >= slot - 2e0 "
    "+ 2e1 and slot - 2e0 <= 1024*floor((224 + slot)/1024) <= 1 + slot - "
    "2e0 + 2e1 and 1024*floor((224 + slot)/1024) <= slot and -24 + 25slot "
    "- 50e0 + e2 <= 25600*floor((224 + slot)/1024) <= 25slot - 50e0 + e2 "
    "and 49slot - 100e0 + 50e1 + 2e2 <= 50176*floor((224 + slot)/1024) <= "
    "1 + 49slot - 100e0 + 50e1 + 2e2) }";

// Filter layout for Conv 6 (1D Conv, 32 out, 24 in, kW 9, stride 2)
inline constexpr char kLayout24Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e0, e1, e2, e4: 2048e4 = 50i0 "
    "- 57i1 - i2 + ct + slot - 2e2 and 0 <= i0 <= 31 and 0 <= i1 <= 23 "
    "and 0 <= i2 <= 8 and 0 <= ct <= 1023 and 0 <= slot <= 4095 and e1 <= "
    "15 and e2 >= 25i0 and 0 <= e2 <= 799 and e2 <= 24 + 25i0 and 50i0 - "
    "i2 <= 2e2 <= 56 + 50i0 - i2 and 1024*floor((224 + slot)/1024) >= slot "
    "- 2e0 + 2e1 and slot - 2e0 <= 1024*floor((224 + slot)/1024) <= 1 + "
    "slot - 2e0 + 2e1 and 1024*floor((224 + slot)/1024) <= slot and -24 + "
    "25slot - 50e0 + e2 <= 25600*floor((224 + slot)/1024) <= 25slot - 50e0 "
    "+ e2 and 49slot - 100e0 + 50e1 + 2e2 <= "
    "50176*floor((224 + slot)/1024) <= 1 + 49slot - 100e0 + 50e1 + 2e2) }";

// Filter layout for Conv 7 (1D Conv, 32 out, 32 in, kW 9)
inline constexpr char kLayout27Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : (1024 + 25i0 - 33i1 - i2 + ct + "
    "1024*floor((-25 - 25i0 + slot)/1024)) mod 2048 = 0 and 0 <= i0 <= 31 "
    "and 0 <= i1 <= 31 and 0 <= i2 <= 8 and 0 <= ct <= 1023 and 0 <= slot "
    "<= 4095 and 1024*floor((-25 - 25i0 + slot)/1024) >= -1823 + slot and "
    "1024*floor((-25 - 25i0 + slot)/1024) >= -1056 - 25i0 + i2 + slot and "
    "1024*floor((-25 - 25i0 + slot)/1024) <= -1024 - 25i0 + slot and "
    "1024*floor((-25 - 25i0 + slot)/1024) <= -1024 - 25i0 + i2 + slot and "
    "1024*floor((-25 - 25i0 + slot)/1024) <= -1024 + slot }";

// Filter layout for Conv 8 (1D Conv, 48 out, 32 in, kW 1, stride 2)
inline constexpr char kLayout29Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e0, e1, e2, e4: i2 = 0 and "
    "1024e4 = 26i0 - 25i1 + ct + slot - 2e2 and 0 <= i0 <= 47 and 0 <= i1 "
    "<= 31 and 0 <= ct <= 1023 and 0 <= slot <= 4095 and e1 <= 23 and e2 "
    ">= 13i0 and 0 <= e2 <= 623 and e2 <= 12 + 13i0 and 26i0 - 25i1 <= 2e2 "
    "<= 799 + 26i0 - 25i1 and 1024*floor((400 + slot)/1024) >= slot - 2e0 "
    "+ 2e1 and slot - 2e0 <= 1024*floor((400 + slot)/1024) <= 1 + slot - "
    "2e0 + 2e1 and 1024*floor((400 + slot)/1024) <= slot and -12 + 13slot "
    "- 26e0 + e2 <= 13312*floor((400 + slot)/1024) <= 13slot - 26e0 + e2 "
    "and 25slot - 52e0 + 26e1 + 2e2 <= 25600*floor((400 + slot)/1024) <= "
    "1 + 25slot - 52e0 + 26e1 + 2e2) }";

// Filter layout for Conv 9 (1D Conv, 48 out, 32 in, kW 9, stride 2)
inline constexpr char kLayout34Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : exists (e0, e1, e2, e4: 2048e4 = 26i0 "
    "- 33i1 - i2 + ct + slot - 2e2 and 0 <= i0 <= 47 and 0 <= i1 <= 31 "
    "and 0 <= i2 <= 8 and 0 <= ct <= 1023 and 0 <= slot <= 4095 and e1 <= "
    "23 and e2 >= 13i0 and 0 <= e2 <= 623 and e2 <= 12 + 13i0 and 26i0 - "
    "i2 <= 2e2 <= 32 + 26i0 - i2 and 1024*floor((400 + slot)/1024) >= slot "
    "- 2e0 + 2e1 and slot - 2e0 <= 1024*floor((400 + slot)/1024) <= 1 + "
    "slot - 2e0 + 2e1 and 1024*floor((400 + slot)/1024) <= slot and -12 + "
    "13slot - 26e0 + e2 <= 13312*floor((400 + slot)/1024) <= 13slot - 26e0 "
    "+ e2 and 25slot - 52e0 + 26e1 + 2e2 <= "
    "25600*floor((400 + slot)/1024) <= 1 + 25slot - 52e0 + 26e1 + 2e2) }";

// Filter layout for Conv 10 (1D Conv, 48 out, 48 in, kW 9)
inline constexpr char kLayout37Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : (13i0 - 21i1 - i2 + ct) mod 1024 = 0 "
    "and 0 <= i0 <= 47 and 0 <= i1 <= 47 and 0 <= i2 <= 8 and 0 <= ct <= "
    "1023 and 0 <= slot <= 4095 and 1024*floor((-13 - 13i0 + slot)/1024) "
    ">= -1647 + slot and 1024*floor((-13 - 13i0 + slot)/1024) >= -1044 - "
    "13i0 + i2 + slot and 1024*floor((-13 - 13i0 + slot)/1024) <= -1024 - "
    "13i0 + slot and 1024*floor((-13 - 13i0 + slot)/1024) <= -1024 - 13i0 "
    "+ i2 + slot and 1024*floor((-13 - 13i0 + slot)/1024) <= -1024 + slot }";

// Filter layout for Conv 11 (1D Conv, 48 out, 48 in, kW 13)
inline constexpr char kLayout39Relation[] =
    "{ [i0, i1, i2] -> [ct, slot] : (i0 - 13i1 - i2 + ct) mod 64 = 0 and "
    "(-13i1 - i2 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 47 and 0 <= i1 "
    "<= 47 and 0 <= i2 <= 12 and 0 <= ct <= 63 and 0 <= slot <= 4095 }";

}  // namespace heir
}  // namespace mlir

#endif  // BENCHMARK_ISL_RELATIONS_H_
