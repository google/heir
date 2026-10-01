package extconst

import (
	"testing"
)

func TestExtConst(t *testing.T) {
	res := Test_fn()
	t.Logf("Result: %v", res)
	if len(res) != 4 || res[0] != 1 || res[1] != 2 || res[2] != 3 || res[3] != 4 {
		t.Errorf("Expected [1 2 3 4], got %v", res)
	}
}
