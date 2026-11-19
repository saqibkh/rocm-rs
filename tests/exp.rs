use rocm_rs::hip::{Device, DeviceMemory, Module};

// B: load an empty code object (what check_sorted does with gpu-sort off)
#[test]
fn b_load_empty_module() {
    let _d = Device::current().unwrap();
    let r = Module::load_data(&[] as &[u8]);
    println!("B result is_err={}", r.is_err());
}

// B2: load a small non-empty garbage buffer
#[test]
fn b2_load_garbage_module() {
    let _d = Device::current().unwrap();
    let r = Module::load_data(&[0u8; 64][..]);
    println!("B2 result is_err={}", r.is_err());
}

// C: async copy, host Vec kept alive until after synchronize (cloned, so the moved one is still dropped)
#[test]
fn c_async_copy_vec_alive() {
    let d = Device::current().unwrap();
    let s = d.get_stream().unwrap();
    let arr: Vec<i32> = vec![1, 3, 2, 4, 5, 6, 8, 7];
    let keep = arr.clone();
    let mem = DeviceMemory::<i32>::new(arr.len()).unwrap();
    mem.copy_from_host_async(arr, &s).unwrap();
    s.synchronize().unwrap();
    let mut back = vec![0i32; 8];
    mem.copy_to_host(&mut back).unwrap();
    println!("C back={:?} keep={:?}", back, keep);
    assert_eq!(back, keep);
}

// D: same as the real test minus the module load: moved Vec is dropped right away, then
// heap churn, then sync + read back
#[test]
fn d_async_copy_vec_dropped() {
    let d = Device::current().unwrap();
    let s = d.get_stream().unwrap();
    let arr: Vec<i32> = vec![1, 3, 2, 4, 5, 6, 8, 7];
    let want = arr.clone();
    let mem = DeviceMemory::<i32>::new(arr.len()).unwrap();
    mem.copy_from_host_async(arr, &s).unwrap();
    let churn: Vec<Vec<i32>> = (0..64).map(|i| vec![i; 8]).collect();
    s.synchronize().unwrap();
    let mut back = vec![0i32; 8];
    mem.copy_to_host(&mut back).unwrap();
    println!("D back={:?} churn_len={}", back, churn.len());
    assert_eq!(back, want);
}
