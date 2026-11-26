//! Diagnostic: print the descriptor-table metadata of a dumped title shader.
//! Run with `SHADER=<dump.bin> cargo test -p victory-shader --test inspect -- --ignored --nocapture`.

#[test]
#[ignore = "diagnostic; set SHADER to a dumped container"]
fn inspect_dump() {
    let path = std::env::var("SHADER").expect("SHADER");
    let bytes = std::fs::read(&path).expect("read shader");
    let shader = victory_shader::emit::translate(&bytes).expect("translate");
    println!(
        "{path}: table_entries={:?} table_dynamic={:?} heap={} memory={} views={} samplers={}",
        shader.table_entries,
        shader.table_dynamic,
        shader.heap,
        shader.memory,
        shader.root_srvs.len() + shader.root_uavs.len() + shader.root_cbvs.len(),
        shader.samplers.len()
    );
}
