// RUN: heir-opt --split-input-file --verify-diagnostics %s

// expected-error @+1 {{logN must be positive, got 0}}
#invalid_log_n = #cheddar.parameter_set<logN = 0, logScale = 35, mainPrimes = [1], auxPrimes = [2]>

// -----

// expected-error @+1 {{logScale must be positive, got -1}}
#invalid_log_scale = #cheddar.parameter_set<logN = 16, logScale = -1, mainPrimes = [1], auxPrimes = [2]>

// -----

// expected-error @+1 {{mainPrimes must not be empty}}
#invalid_main_primes = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [], auxPrimes = [2]>

// -----

// expected-error @+1 {{auxPrimes must not be empty}}
#invalid_aux_primes = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1], auxPrimes = []>

// -----

// expected-error @+1 {{levelConfig size must be even, got 3}}
#invalid_level_config_odd = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1, 2], auxPrimes = [3], levelConfig = [1, 0, 2]>

// -----

// expected-error @+1 {{numMain at level 0 (5) must be in range [0, 2]}}
#invalid_level_config_bounds = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1, 2], auxPrimes = [3], levelConfig = [5, 0]>

// -----

// expected-error @+1 {{additionalBase must have exactly 2 elements, got 1}}
#invalid_additional_base = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1], auxPrimes = [2], additionalBase = [0]>

// -----

// expected-error @+1 {{wordBits must be 32 or 64, got 16}}
#invalid_word_bits = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1], auxPrimes = [2], wordBits = 16>

// -----

// expected-error @+1 {{numCtsLevels is required when boot is true}}
#invalid_boot_missing_cts = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1], auxPrimes = [2], boot = true, numStcLevels = 3>

// -----

// expected-error @+1 {{numStcLevels is required when boot is true}}
#invalid_boot_missing_stc = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1], auxPrimes = [2], boot = true, numCtsLevels = 4>

// -----

// expected-error @+1 {{defaultEncryptionLevel (2) must be in range [0, 2) given levelConfig with 2 levels}}
#invalid_default_encryption_level = #cheddar.parameter_set<logN = 16, logScale = 35, mainPrimes = [1, 2], auxPrimes = [3], levelConfig = [1, 0, 2, 0], defaultEncryptionLevel = 2>
