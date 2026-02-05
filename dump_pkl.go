package main

import (
	"archive/zip"
	"fmt"
	"path"

	"github.com/nlpodyssey/gopickle/pickle"
	"github.com/nlpodyssey/gopickle/types"
)

func main() {
	filename := "tests/Examples/common/mnist/data/traced_model.pt"
	r, err := zip.OpenReader(filename)
	if err != nil {
		panic(err)
	}
	defer r.Close()

	var dataPkl *zip.File
	for _, f := range r.File {
		_, recordName := path.Split(f.Name)
		if recordName == "data.pkl" {
			dataPkl = f
			break
		}
	}

	if dataPkl == nil {
		panic("data.pkl not found")
	}

	rc, err := dataPkl.Open()
	if err != nil {
		panic(err)
	}
	defer rc.Close()

	u := pickle.NewUnpickler(rc)
	u.PersistentLoad = func(savedId interface{}) (interface{}, error) {
		return nil, nil
	}
	u.FindClass = func(module, name string) (interface{}, error) {
		return nil, nil
	}

	res, err := u.Load()
	if err != nil {
		fmt.Printf("Error loading: %v\n", err)
	}

	fmt.Printf("Result type: %T\n", res)
	if dict, ok := res.(*types.Dict); ok {
		for _, entry := range *dict {
			fmt.Printf("Key: %v\n", entry.Key)
		}
	} else if list, ok := res.(*types.List); ok {
		fmt.Printf("List length: %d\n", len(*list))
		for i, item := range *list {
			fmt.Printf("Item %d: %T\n", i, item)
		}
	} else if tuple, ok := res.(*types.Tuple); ok {
		fmt.Printf("Tuple length: %d\n", tuple.Len())
		for i := 0; i < tuple.Len(); i++ {
			fmt.Printf("Item %d: %T\n", i, tuple.Get(i))
		}
	}
}
