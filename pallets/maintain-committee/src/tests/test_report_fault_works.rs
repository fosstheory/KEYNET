use crate::types::{MTOrderStatus, ReportStatus};

use super::super::{mock::*, ReporterStakeInfo};
use frame_support::assert_ok;
use std::convert::TryInto;

// 报告机器被租用，但是无法访问
// case1: 只有1委员会预订，同意报告
// case2: 只有1委员会预订，拒绝报告
// case3: 只有1人预订，提交了Hash,未提交最终结果
// case3: 只有1人预订，未提交Hash,未提交最终结果

// 报告机器被租用，但是无法访问: 只有一个人预订，10分钟后检查结果，两天后结果执行
#[test]
fn report_machine_inaccessible_works1() {
    new_test_with_init_params_ext().execute_with(|| {
        let committee: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::One).into();
        let reporter: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Two).into();
        let machine_id = "8eaf04151687736326c9fea17e25fc5287613693c912909cb226aa4794f26a48".as_bytes().to_vec();
        let _machine_stash: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Ferdie).into();

        // 记录：ReportInfo, LiveReport, ReporterReport 并支付处理所需的金额
        assert_ok!(MaintainCommittee::report_machine_fault(
            Origin::signed(reporter),
            crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
        ));

        // 判断调用举报之后的状态
        {
            assert_eq!(
                &MaintainCommittee::live_report(),
                &crate::MTLiveReportList { bookable_report: vec![0], ..Default::default() }
            );
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    machine_id: machine_id.clone(),
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: crate::ReportStatus::Reported,

                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::reporter_report(&reporter),
                &crate::ReporterReportList { processing_report: vec![0], ..Default::default() }
            );
            // TODO: 检查free_balance
        }

        // 委员会订阅机器故障报告
        assert_ok!(MaintainCommittee::committee_book_report(Origin::signed(committee), 0));

        // 检查订阅之后的状态
        // do_report_machine_fault:
        // - Writes:
        // LiveReport, ReportInfo, CommitteeOps, CommitteeOrder, committee pay txFee
        {
            assert_eq!(
                &MaintainCommittee::live_report(),
                &crate::MTLiveReportList { bookable_report: vec![0], ..Default::default() }
            );
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::WaitingBook,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    booked_time: 11,
                    order_status: MTOrderStatus::Verifying,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_order(&committee),
                &crate::MTCommitteeOrderList { booked_report: vec![0], ..Default::default() }
            );

            assert_eq!(Balances::free_balance(&committee), INIT_BALANCE - 20000 * ONE_DBC - 10 * ONE_DBC);
        }

        // 委员会首先提交Hash: 内容为 订单ID + 验证人自己的随机数 + 机器是否有问题
        // hash(0abcd1) => 0x73124a023f585b4018b9ed3593c7470a
        let offline_committee_hash: [u8; 16] =
            hex::decode("73124a023f585b4018b9ed3593c7470a").unwrap().try_into().unwrap();
        // - Writes:
        // LiveReport, CommitteeOps, CommitteeOrder, ReportInfo
        assert_ok!(MaintainCommittee::committee_submit_verify_hash(
            Origin::signed(committee),
            0,
            offline_committee_hash.clone()
        ));

        // 检查状态
        // TODO: 如果两个同时来预订的状态
        {
            assert_eq!(
                &MaintainCommittee::live_report(),
                &crate::MTLiveReportList { bookable_report: vec![0], ..Default::default() }
            );
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    hashed_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::WaitingBook,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    booked_time: 11,
                    confirm_hash: offline_committee_hash,
                    hash_time: 11,
                    order_status: MTOrderStatus::WaitingRaw,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_order(&committee),
                &crate::MTCommitteeOrderList { booked_report: vec![], hashed_report: vec![0], ..Default::default() }
            );
        }

        run_to_block(21);
        // - Writes:
        // ReportInfo, committee_ops,
        assert_ok!(MaintainCommittee::committee_submit_inaccessible_raw(
            Origin::signed(committee),
            0,
            "abcd".as_bytes().to_vec(),
            true
        ));

        // 检查提交了确认信息后的状态
        {
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    hashed_committee: vec![committee],
                    confirmed_committee: vec![committee],
                    support_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::SubmittingRaw,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    booked_time: 11,
                    confirm_hash: offline_committee_hash,
                    hash_time: 11,
                    confirm_time: 22,
                    confirm_result: true,
                    order_status: MTOrderStatus::Finished,
                    ..Default::default()
                }
            );
        }

        run_to_block(23);

        // 检查summary的结果
        // summary_a_inaccessible
        // - Writes:
        // ReportInfo, ReportResult, CommitteeOrder, CommitteeOps
        // LiveReport, UnhandledReportResult, ReporterReport,
        {
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    hashed_committee: vec![committee],
                    confirmed_committee: vec![committee],
                    support_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::CommitteeConfirmed,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::report_result(0),
                &crate::MTReportResultInfo {
                    report_id: 0,
                    reporter,
                    reporter_stake: 1000 * ONE_DBC,
                    reward_committee: vec![committee],
                    machine_id: machine_id.clone(),
                    slash_time: 22,
                    slash_exec_time: 22 + 2880 * 2,
                    report_result: crate::ReportResultType::ReportSucceed,
                    slash_result: crate::MCSlashResult::Pending,
                    // inconsistent_committee, unruly_committee, machine_stash,
                    // committee_stake
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_order(&committee),
                &crate::MTCommitteeOrderList { finished_report: vec![0], ..Default::default() }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    booked_time: 11,
                    confirm_hash: offline_committee_hash,
                    hash_time: 11,
                    confirm_time: 22,
                    confirm_result: true,
                    order_status: crate::MTOrderStatus::Finished,

                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::live_report(),
                &crate::MTLiveReportList { finished_report: vec![0], ..Default::default() }
            );
            let unhandled_report_result: Vec<u64> = vec![0];
            assert_eq!(&MaintainCommittee::unhandled_report_result(), &unhandled_report_result);
            assert_eq!(
                &MaintainCommittee::reporter_report(&reporter),
                &crate::ReporterReportList { succeed_report: vec![0], ..Default::default() }
            );
        }

        // TODO: 两天后，根据结果进行惩罚
        // TODO: 机器在举报成功后会立即被下线
    })
}

#[test]
fn report_machine_inaccessible_works2() {
    new_test_with_init_params_ext().execute_with(|| {
        let committee: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::One).into();
        let reporter: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Two).into();
        let machine_id = "8eaf04151687736326c9fea17e25fc5287613693c912909cb226aa4794f26a48".as_bytes().to_vec();
        let _machine_stash: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Ferdie).into();

        // 记录：ReportInfo, LiveReport, ReporterReport 并支付处理所需的金额
        assert_ok!(MaintainCommittee::report_machine_fault(
            Origin::signed(reporter),
            crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
        ));

        // 委员会订阅机器故障报告
        assert_ok!(MaintainCommittee::committee_book_report(Origin::signed(committee), 0));

        // 委员会首先提交Hash: 内容为 订单ID + 验证人自己的随机数 + 机器是否有问题
        // hash(0abcd1) => 0x73124a023f585b4018b9ed3593c7470a
        let offline_committee_hash: [u8; 16] =
            hex::decode("98b18d58d8d3bc2f2037cb8310dd6f0e").unwrap().try_into().unwrap();
        // - Writes:
        // LiveReport, CommitteeOps, CommitteeOrder, ReportInfo
        assert_ok!(MaintainCommittee::committee_submit_verify_hash(
            Origin::signed(committee),
            0,
            offline_committee_hash.clone()
        ));

        run_to_block(21);
        // - Writes:
        // ReportInfo, committee_ops,
        assert_ok!(MaintainCommittee::committee_submit_inaccessible_raw(
            Origin::signed(committee),
            0,
            "fedcba111".as_bytes().to_vec(),
            false
        ));

        // 检查提交了确认信息后的状态
        {
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    hashed_committee: vec![committee],
                    confirmed_committee: vec![committee],
                    // support_committee: vec![committee],
                    against_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::SubmittingRaw,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    booked_time: 11,
                    confirm_hash: offline_committee_hash,
                    hash_time: 11,
                    confirm_time: 22,
                    confirm_result: false,
                    order_status: MTOrderStatus::Finished,
                    ..Default::default()
                }
            );
        }

        run_to_block(23);

        // 检查summary的结果
        // summary_a_inaccessible
        // - Writes:
        // ReportInfo, ReportResult, CommitteeOrder, CommitteeOps
        // LiveReport, UnhandledReportResult, ReporterReport,
        {
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    hashed_committee: vec![committee],
                    confirmed_committee: vec![committee],
                    // support_committee: vec![committee],
                    against_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::CommitteeConfirmed,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::report_result(0),
                &crate::MTReportResultInfo {
                    report_id: 0,
                    reporter,
                    reporter_stake: 1000 * ONE_DBC,
                    reward_committee: vec![committee],
                    machine_id: machine_id.clone(),
                    slash_time: 22,
                    slash_exec_time: 22 + 2880 * 2,
                    report_result: crate::ReportResultType::ReportRefused,
                    slash_result: crate::MCSlashResult::Pending,
                    // inconsistent_committee, unruly_committee, machine_stash,
                    // committee_stake,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_order(&committee),
                &crate::MTCommitteeOrderList { finished_report: vec![0], ..Default::default() }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    booked_time: 11,
                    confirm_hash: offline_committee_hash,
                    hash_time: 11,
                    confirm_time: 22,
                    confirm_result: false,
                    order_status: crate::MTOrderStatus::Finished,

                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::live_report(),
                &crate::MTLiveReportList { finished_report: vec![0], ..Default::default() }
            );
            let unhandled_report_result: Vec<u64> = vec![0];
            assert_eq!(&MaintainCommittee::unhandled_report_result(), &unhandled_report_result);
            assert_eq!(
                &MaintainCommittee::reporter_report(&reporter),
                &crate::ReporterReportList { failed_report: vec![0], ..Default::default() }
            );
        }

        // TODO: 两天后，根据结果进行惩罚
        // TODO: 机器在举报成功后会立即被下线
    })
}

#[test]
fn report_machine_inaccessible_works3() {
    new_test_with_init_params_ext().execute_with(|| {
        let committee: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::One).into();
        let reporter: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Two).into();
        let machine_id = "8eaf04151687736326c9fea17e25fc5287613693c912909cb226aa4794f26a48".as_bytes().to_vec();
        let _machine_stash: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Ferdie).into();

        // 记录：ReportInfo, LiveReport, ReporterReport 并支付处理所需的金额
        assert_ok!(MaintainCommittee::report_machine_fault(
            Origin::signed(reporter),
            crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
        ));

        // 委员会订阅机器故障报告
        assert_ok!(MaintainCommittee::committee_book_report(Origin::signed(committee), 0));

        // 委员会首先提交Hash: 内容为 订单ID + 验证人自己的随机数 + 机器是否有问题
        // hash(0abcd1) => 0x73124a023f585b4018b9ed3593c7470a
        let offline_committee_hash: [u8; 16] =
            hex::decode("98b18d58d8d3bc2f2037cb8310dd6f0e").unwrap().try_into().unwrap();
        // - Writes:
        // LiveReport, CommitteeOps, CommitteeOrder, ReportInfo
        assert_ok!(MaintainCommittee::committee_submit_verify_hash(
            Origin::signed(committee),
            0,
            offline_committee_hash.clone()
        ));

        run_to_block(32);

        // 检查summary的结果

        // 检查 report_id: 0

        // summary_a_inaccessible
        // - Writes:
        // ReportInfo, ReportResult, CommitteeOrder, CommitteeOps
        // LiveReport, UnhandledReportResult, ReporterReport,
        {
            assert_eq!(
                &MaintainCommittee::report_info(0),
                &crate::MTReportInfoDetail {
                    reporter,
                    report_time: 11,
                    reporter_stake: 1000 * ONE_DBC,
                    first_book_time: 11,
                    machine_id: machine_id.clone(),
                    verifying_committee: None,
                    booked_committee: vec![committee],
                    hashed_committee: vec![committee],
                    // confirmed_committee: vec![],
                    // support_committee: vec![committee],
                    // against_committee: vec![committee],
                    confirm_start: 11 + 10,
                    machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
                    report_status: ReportStatus::SubmittingRaw,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::report_result(0),
                &crate::MTReportResultInfo {
                    report_id: 0,
                    reporter,
                    reporter_stake: 1000 * ONE_DBC,
                    unruly_committee: vec![committee],
                    machine_id: machine_id.clone(),
                    slash_time: 32,
                    slash_exec_time: 32 + 2880 * 2,
                    report_result: crate::ReportResultType::NoConsensus,
                    slash_result: crate::MCSlashResult::Pending,
                    // inconsistent_committee, reward_committee, machine_stash,
                    // committee_stake,
                    ..Default::default()
                }
            );
            assert_eq!(
                &MaintainCommittee::committee_order(&committee),
                &crate::MTCommitteeOrderList { ..Default::default() }
            );
            assert_eq!(
                &MaintainCommittee::committee_ops(&committee, 0),
                &crate::MTCommitteeOpsDetail {
                    // booked_time: 11,
                    // confirm_result: false,
                    // order_status: crate::MTOrderStatus::Finished,
                    ..Default::default()
                }
            );
            assert_eq!(&MaintainCommittee::live_report(), &crate::MTLiveReportList { ..Default::default() });
            let unhandled_report_result: Vec<u64> = vec![0];
            assert_eq!(&MaintainCommittee::unhandled_report_result(), &unhandled_report_result);
            assert_eq!(
                &MaintainCommittee::reporter_report(&reporter),
                &crate::ReporterReportList { failed_report: vec![0], ..Default::default() }
            );
        }

        // TODO: 检查 report_id: 1
        // {
        //     assert_eq!(
        //         &MaintainCommittee::report_info(1),
        //         &crate::MTReportInfoDetail {
        //             reporter,
        //             report_time: 11,
        //             // reporter_stake: 1000 * ONE_DBC,
        //             first_book_time: 11,
        //             machine_id: machine_id.clone(),
        //             verifying_committee: None,
        //             booked_committee: vec![committee],
        //             hashed_committee: vec![committee],
        //             // confirmed_committee: vec![],
        //             // support_committee: vec![committee],
        //             // against_committee: vec![committee],
        //             confirm_start: 11 + 10,
        //             machine_fault_type: crate::MachineFaultType::RentedInaccessible(machine_id.clone()),
        //             report_status: ReportStatus::CommitteeConfirmed,
        //             ..Default::default()
        //         }
        //     );
        //     assert_eq!(
        //         &MaintainCommittee::report_result(1),
        //         &crate::MTReportResultInfo {
        //             report_id: 0,
        //             reporter,
        //             reward_committee: vec![committee],
        //             machine_id: machine_id.clone(),
        //             slash_time: 22,
        //             slash_exec_time: 22 + 2880 * 2,
        //             report_result: crate::ReportResultType::ReportRefused,
        //             slash_result: crate::MCSlashResult::Pending,
        //             // inconsistent_committee, unruly_committee, machine_stash,
        //             // committee_stake, reporter_stake
        //             ..Default::default()
        //         }
        //     );
        //     assert_eq!(
        //         &MaintainCommittee::committee_order(&committee),
        //         &crate::MTCommitteeOrderList { finished_report: vec![0], ..Default::default() }
        //     );
        //     assert_eq!(
        //         &MaintainCommittee::committee_ops(&committee, 1),
        //         &crate::MTCommitteeOpsDetail {
        //             booked_time: 11,
        //             confirm_hash: offline_committee_hash,
        //             hash_time: 11,
        //             confirm_time: 22,
        //             confirm_result: false,
        //             order_status: crate::MTOrderStatus::Finished,

        //             ..Default::default()
        //         }
        //     );
        //     assert_eq!(
        //         &MaintainCommittee::live_report(),
        //         &crate::MTLiveReportList { finished_report: vec![0], ..Default::default() }
        //     );
        //     let unhandled_report_result: Vec<u64> = vec![0];
        //     assert_eq!(&MaintainCommittee::unhandled_report_result(), &unhandled_report_result);
        //     assert_eq!(
        //         &MaintainCommittee::reporter_report(&reporter),
        //         &crate::ReporterReportList { failed_report: vec![0], ..Default::default() }
        //     );
        // }

        // TODO: 两天后，根据结果进行惩罚
        // TODO: 机器在举报成功后会立即被下线
    })
}

// 报告其他类型的错误
#[test]
fn report_machine_fault_works() {
    new_test_with_init_params_ext().execute_with(|| {
        let controller: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Eve).into();
        let committee1: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::One).into();

        // 报告人
        let reporter: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Two).into();
        // 报告人解密pubkey
        let reporter_boxpubkey = hex::decode("1e71b5a83ccdeff1592062a1d4da4a272691f08e2024a1ca75a81d534a76210a")
            .unwrap()
            .try_into()
            .unwrap();

        let report_hash: [u8; 16] = hex::decode("986fffc16e63d3f7c43fe1a272ba3ba1").unwrap().try_into().unwrap();

        let machine_id = "8eaf04151687736326c9fea17e25fc5287613693c912909cb226aa4794f26a48".as_bytes().to_vec();
        let reporter_rand_str = "abcdef".as_bytes().to_vec();
        let committee_rand_str = "fedcba".as_bytes().to_vec();
        let err_reason = "它坏了".as_bytes().to_vec();
        let committee_hash: [u8; 16] = hex::decode("0029f96394d458279bcd0c232365932a").unwrap().try_into().unwrap();

        assert_ok!(MaintainCommittee::report_machine_fault(
            Origin::signed(reporter),
            crate::MachineFaultType::RentedHardwareMalfunction(report_hash, reporter_boxpubkey),
        ));

        // report_machine hardware fault:
        // - Writes:
        // ReporterStake, ReportInfo, LiveReport, ReporterReport
        let report_status = crate::MTReportInfoDetail {
            reporter,
            report_time: 11,
            reporter_stake: 1000 * ONE_DBC, // 15,000,000 / 12,000
            machine_fault_type: crate::MachineFaultType::RentedHardwareMalfunction(report_hash, reporter_boxpubkey),
            ..Default::default()
        };
        assert_eq!(&MaintainCommittee::report_info(0), &report_status);
        assert_eq!(
            &MaintainCommittee::reporter_stake(&reporter),
            &ReporterStakeInfo {
                staked_amount: 20000 * ONE_DBC,
                used_stake: 1000 * ONE_DBC,
                can_claim_reward: 0,
                claimed_reward: 0,
            }
        );
        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { bookable_report: vec![0], ..Default::default() }
        );
        assert_eq!(
            &MaintainCommittee::reporter_report(&reporter),
            &crate::ReporterReportList { processing_report: vec![0], ..Default::default() }
        );

        // 委员会订阅机器故障报告
        assert_ok!(MaintainCommittee::committee_book_report(Origin::signed(committee1), 0));

        // book_fault_order:
        // - Writes:
        // LiveReport, ReportInfo, CommitteeOps, CommitteeOrder
        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { verifying_report: vec![0], ..Default::default() }
        );
        let mut report_info = crate::MTReportInfoDetail {
            first_book_time: 11,
            verifying_committee: Some(committee1.clone()),
            booked_committee: vec![committee1.clone()],
            confirm_start: 11 + 360,
            report_status: crate::ReportStatus::Verifying,
            ..report_status
        };
        assert_eq!(&MaintainCommittee::report_info(0), &report_info);
        let mut committee_ops = crate::MTCommitteeOpsDetail {
            booked_time: 11,
            staked_balance: 1000 * ONE_DBC,
            order_status: crate::MTOrderStatus::WaitingEncrypt,
            ..Default::default()
        };
        assert_eq!(&MaintainCommittee::committee_ops(&committee1, 0), &committee_ops);
        assert_eq!(
            &MaintainCommittee::committee_order(&committee1),
            &crate::MTCommitteeOrderList { booked_report: vec![0], ..Default::default() }
        );

        // 提交加密信息
        let encrypted_err_info: Vec<u8> = hex::decode("01405deeef2a8b0f4a09380d14431dd10fde1ad62b3c27b3fbea4701311d")
            .unwrap()
            .try_into()
            .unwrap();
        assert_ok!(MaintainCommittee::reporter_add_encrypted_error_info(
            Origin::signed(reporter),
            0,
            committee1,
            encrypted_err_info.clone()
        ));

        // add_encrypted_err_info:
        // - Writes:
        // CommitteeOps, ReportInfo

        report_info.get_encrypted_info_committee.push(committee1);
        assert_eq!(&MaintainCommittee::report_info(0), &report_info);
        committee_ops.encrypted_err_info = Some(encrypted_err_info.clone());
        committee_ops.encrypted_time = 11;
        committee_ops.order_status = crate::MTOrderStatus::Verifying;

        assert_eq!(&MaintainCommittee::committee_ops(&committee1, 0), &committee_ops);

        // 提交验证Hash
        assert_ok!(MaintainCommittee::committee_submit_verify_hash(
            Origin::signed(committee1),
            0,
            committee_hash.clone()
        ));

        // submit_confirm_hash:
        // - Writes:
        // CommitteeOrder, CommitteeOps, ReportInfo, LiveReport

        report_info.verifying_committee = None;
        report_info.hashed_committee.push(committee1);
        report_info.report_status = crate::ReportStatus::WaitingBook;
        assert_eq!(&MaintainCommittee::report_info(0), &report_info);
        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { bookable_report: vec![0], ..Default::default() }
        );
        committee_ops.confirm_hash = committee_hash;
        committee_ops.order_status = crate::MTOrderStatus::WaitingRaw;
        committee_ops.hash_time = 11;
        assert_eq!(&MaintainCommittee::committee_ops(&committee1, 0), &committee_ops);
        assert_eq!(
            &MaintainCommittee::committee_order(&committee1),
            &crate::MTCommitteeOrderList { hashed_report: vec![0], ..Default::default() }
        );

        // 3个小时之后才能提交：
        run_to_block(360 + 13);

        report_info.report_status = crate::ReportStatus::SubmittingRaw;
        assert_eq!(&MaintainCommittee::report_info(0), &report_info);
        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { waiting_raw_report: vec![0], ..Default::default() }
        );

        // submit_confirm_raw:
        // - Writes:
        // ReportInfo, CommitteeOps
        let extra_err_info = Vec::new();
        assert_ok!(MaintainCommittee::committee_submit_verify_raw(
            Origin::signed(committee1),
            0,
            machine_id.clone(),
            reporter_rand_str,
            committee_rand_str,
            err_reason.clone(),
            extra_err_info,
            true
        ));

        report_info.confirmed_committee = vec![committee1.clone()];
        report_info.support_committee = vec![committee1.clone()];
        report_info.machine_id = machine_id.clone();
        report_info.err_info = err_reason;
        assert_eq!(&MaintainCommittee::report_info(0), &report_info);

        committee_ops.confirm_time = 374;
        committee_ops.confirm_result = true;
        committee_ops.order_status = crate::MTOrderStatus::Finished;

        assert_eq!(&MaintainCommittee::committee_ops(&committee1, 0), &committee_ops);

        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { waiting_raw_report: vec![0], ..Default::default() }
        );

        assert!(match MaintainCommittee::summary_report(0) {
            crate::ReportConfirmStatus::Confirmed(..) => true,
            _ => false,
        });

        // assert_eq!(&super::ReportConfirmStatus::Confirmed(_, _, _), MaintainCommittee::summary_report(0));

        run_to_block(360 + 14);

        // summary_fault_case -> summary_waiting_raw -> Confirmed -> mt_machine_offline
        // - Writes:
        // committee_stake; committee_order; LiveReport;
        // report_info.report_status = super::ReportStatus::CommitteeConfirmed;
        assert_eq!(Committee::committee_stake(committee1).used_stake, 0);
        assert_eq!(
            MaintainCommittee::committee_order(committee1),
            crate::MTCommitteeOrderList { finished_report: vec![0], ..Default::default() }
        );
        // assert_eq!(&MachineCommittee::report_info(0), &super::MTReportInfoDetail { ..Default::default() });
        // assert_eq!(&MaintainCommittee::report_info(0), &report_info);
        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { finished_report: vec![0], ..Default::default() }
        );

        // mt_machine_offline -> machine_offline
        // - Writes:
        // MachineInfo, LiveMachine, current_era_stash_snap, next_era_stash_snap, current_era_machine_snap, next_era_machine_snap
        // SysInfo, SatshMachine, PosGPUInfo

        assert_eq!(
            &MaintainCommittee::live_report(),
            &crate::MTLiveReportList { finished_report: vec![0], ..Default::default() }
        );

        run_to_block(2880 + 400);

        // 报告人上线机器
        assert_ok!(OnlineProfile::controller_report_online(Origin::signed(controller), machine_id.clone()));
    })
}
