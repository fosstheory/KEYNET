use codec::{alloc::string::ToString, Decode, Encode, HasCompact};
#[cfg(feature = "std")]
use serde::{Deserialize, Serialize};
use sp_io::hashing::blake2_128;
use sp_runtime::{Perbill, RuntimeDebug};
use sp_std::{collections::btree_map::BTreeMap, prelude::*};

pub type MachineId = Vec<u8>;
pub type EraIndex = u32;
pub type ImageName = Vec<u8>;

pub const LOCK_BLOCK_EXPIRATION: u32 = 3; // in block number

#[derive(Debug, PartialEq, Eq, Clone, Encode, Decode, Default)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub struct MachineInfoDetail {
    pub committee_upload_info: CommitteeUploadInfo,
    pub staker_customize_info: StakerCustomizeInfo,
}

#[derive(Debug, PartialEq, Eq, Clone, Encode, Decode, Default)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub struct CommitteeUploadInfo {
    pub machine_id: MachineId,
    pub gpu_type: Vec<u8>, // GPU型号
    pub gpu_num: u32,      // GPU数量
    pub cuda_core: u32,    // CUDA core数量
    pub gpu_mem: u64,      // GPU显存
    pub calc_point: u64,   // 算力值
    pub sys_disk: u64,     // 系统盘大小
    pub data_disk: u64,    // 数据盘大小
    pub cpu_type: Vec<u8>, // CPU型号
    pub cpu_core_num: u32, // CPU内核数
    pub cpu_rate: u64,     // CPU频率
    pub mem_num: u64,      // 内存数

    pub rand_str: Vec<u8>,
    pub is_support: bool, // 委员会是否支持该机器上线
}

impl CommitteeUploadInfo {
    pub fn hash(&self) -> [u8; 16] {
        let gpu_num: Vec<u8> = self.gpu_num.to_string().into();
        let cuda_core: Vec<u8> = self.cuda_core.to_string().into();
        let gpu_mem: Vec<u8> = self.gpu_mem.to_string().into();
        let calc_point: Vec<u8> = self.calc_point.to_string().into();
        let sys_disk: Vec<u8> = self.sys_disk.to_string().into();
        let data_disk: Vec<u8> = self.data_disk.to_string().into();
        let cpu_core_num: Vec<u8> = self.cpu_core_num.to_string().into();
        let cpu_rate: Vec<u8> = self.cpu_rate.to_string().into();
        let mem_num: Vec<u8> = self.mem_num.to_string().into();

        let is_support: Vec<u8> = if self.is_support { "1".into() } else { "0".into() };

        let mut raw_info = Vec::new();
        raw_info.extend(self.machine_id.clone());
        raw_info.extend(self.gpu_type.clone());
        raw_info.extend(gpu_num);
        raw_info.extend(cuda_core);
        raw_info.extend(gpu_mem);
        raw_info.extend(calc_point);
        raw_info.extend(sys_disk);
        raw_info.extend(data_disk);
        raw_info.extend(self.cpu_type.clone());
        raw_info.extend(cpu_core_num);
        raw_info.extend(cpu_rate);
        raw_info.extend(mem_num);

        raw_info.extend(self.rand_str.clone());
        raw_info.extend(is_support);

        return blake2_128(&raw_info);
    }
}

// 不确定值，由机器管理者提交
#[derive(Debug, PartialEq, Eq, Clone, Encode, Decode)]
#[cfg_attr(feature = "std", derive(Serialize, Deserialize))]
pub struct StakerCustomizeInfo {
    pub left_change_time: u64, // 用户对贷款及经纬度的修改次数

    pub upload_net: u64,   // 上行带宽
    pub download_net: u64, // 下行带宽
    pub longitude: u64,    // 经度
    pub latitude: u64,     // 纬度

    pub images: Vec<ImageName>, // 镜像名称
}

impl Default for StakerCustomizeInfo {
    fn default() -> Self {
        StakerCustomizeInfo {
            left_change_time: 3,
            upload_net: 0,   // 不确定值, 存储平均值
            download_net: 0, // 不确定值, 存储平均值
            longitude: 0,    // 经度, 不确定值，存储平均值
            latitude: 0,     // 纬度, 不确定值，存储平均值
            images: Vec::new(),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Encode, Decode, RuntimeDebug)]
pub struct UnlockChunk<Balance: HasCompact> {
    #[codec(compact)]
    pub value: Balance,

    #[codec(compact)]
    pub era: EraIndex,
}

// 记录每个Era的机器的总分
#[derive(PartialEq, Encode, Decode, Default, RuntimeDebug, Clone)]
pub struct EraMachinePoints<AccountId: Ord> {
    // 所有可以奖励的机器总得分
    pub total: u64,
    // 某个Era，所有的机器的基础得分,机器的在线状态
    pub individual_points: BTreeMap<MachineId, MachineGradeStatus>,
    // 某个Era，用户的得分膨胀系数快照
    pub staker_statistic: BTreeMap<AccountId, StakerStatistics>,
}

#[derive(PartialEq, Encode, Decode, Default, RuntimeDebug, Clone)]
pub struct MachineGradeStatus {
    pub basic_grade: u64,
    pub is_online: bool,
}

#[derive(PartialEq, Encode, Decode, Default, RuntimeDebug, Clone)]
pub struct StakerStatistics {
    pub online_num: u64,               // 用户在线的机器数量
    pub inflation: Perbill,            // 用户对应的膨胀系数
    pub machine_total_calc_point: u64, // 用户的机器的总计算点数得分(不考虑膨胀)
    pub rent_extra_grade: u64,         // 用户机器因被租用获得的额外得分
}
