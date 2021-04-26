use alt_serde::{Deserialize, Deserializer};
use codec::{Decode, Encode};
use sp_std::{prelude::*, str};

pub fn de_string_to_bytes<'de, D>(de: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: &str = Deserialize::deserialize(de)?;
    Ok(s.as_bytes().to_vec())
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct MachineInfo {
    pub error_code: u32,
    pub data: MachineData,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct MachineData {
    pub grades: Grades,

    // 估价，单位：美元*10**6
    pub appraisal_price: u64,

    cpu: CPU,

    #[serde(deserialize_with = "de_string_to_bytes")]
    cpu_usage: Vec<u8>,

    disk: Disk,
    gpu: GPU,
    gpu_state: GPUStatus,
    gpu_usage: Vec<GPUUsage>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    ip: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    mem: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    mem_usage: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    network_dl: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    network_ul: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    os: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    state: Vec<u8>,

    #[serde(deserialize_with = "de_string_to_bytes")]
    version: Vec<u8>,

    pub wallet: Vec<OneWallet>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct OneWallet(#[serde(deserialize_with = "de_string_to_bytes")] pub Vec<u8>);

#[derive(Deserialize, Encode, Decode, Default, Debug, Clone, PartialEq, Copy)]
#[serde(crate = "alt_serde")]
pub struct Grades {
    pub cpu: u64,
    pub disk: u64,
    pub gpu: u64,
    pub mem: u64,
    pub net: u64,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct CPU {
    #[serde(deserialize_with = "de_string_to_bytes")]
    num: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    #[serde(rename = "type")]
    _type: Vec<u8>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct Disk {
    #[serde(deserialize_with = "de_string_to_bytes")]
    size: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    free: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    #[serde(rename = "type")]
    _type: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    speed: Vec<u8>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct GPU {
    #[serde(deserialize_with = "de_string_to_bytes")]
    num: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    driver: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    cuda: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    p2p: Vec<u8>,
    gpus: Vec<GPUDetail>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct GPUDetail {
    #[serde(deserialize_with = "de_string_to_bytes")]
    id: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    #[serde(rename = "type")]
    _type: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    mem: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    pcie_bandwidth: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    mem_bandwidth: Vec<u8>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct GPUStatus {
    gpus: Vec<OneGPUStatus>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct OneGPUStatus {
    #[serde(deserialize_with = "de_string_to_bytes")]
    id: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    state: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    #[serde(rename = "type")]
    _type: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    uuid: Vec<u8>,
}

#[derive(Deserialize, Encode, Decode, Default, Debug)]
#[serde(crate = "alt_serde")]
pub struct GPUUsage {
    #[serde(deserialize_with = "de_string_to_bytes")]
    gpu: Vec<u8>,
    #[serde(deserialize_with = "de_string_to_bytes")]
    mem: Vec<u8>,
}