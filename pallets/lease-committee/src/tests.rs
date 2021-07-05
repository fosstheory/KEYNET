use crate::{mock::*, LCMachineCommitteeList};
use committee::CommitteeList;
use dbc_price_ocw::MAX_LEN;
use frame_support::assert_ok;
use online_profile::{LiveMachine, StakerCustomizeInfo};
use std::convert::TryInto;

#[test]
#[rustfmt::skip]
fn set_default_value_works() {
    new_test_ext().execute_with(|| {
        System::set_block_number(1); // 随机函数需要

        let alice: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Alice).into();
        let bob: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Bob).into();
        let charile: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Charlie).into();
        let dave: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Dave).into();

        let one: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::One).into();
        let two: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Two).into();

        let controller: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Eve).into(); // Controller
        let stash: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Ferdie).into(); // Stash
        let machine_id = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"; // Bob account

        assert_eq!(Balances::free_balance(alice), 1000_000);

        // 初始化price_ocw (0.012$)
        assert_eq!(DBCPriceOCW::avg_price(), None);
        for i in 0..MAX_LEN {
            DBCPriceOCW::add_price(12_000u64);
        }
        DBCPriceOCW::add_avg_price();
        assert_eq!(DBCPriceOCW::avg_price(), Some(12_000u64));

        // 初始化设置参数
        // 委员会每次抢单质押数量 (15$)
        assert_ok!(Committee::set_staked_usd_per_order(RawOrigin::Root.into(), 15_000_000u32.into()));
        // 操作时的固定费率
        assert_ok!(GenericFunc::set_fixed_tx_fee(RawOrigin::Root.into(), 10u32.into()));
        // 每张GPU质押数量
        assert_ok!(OnlineProfile::set_gpu_stake(RawOrigin::Root.into(), 100_000u32.into()));
        // 设置奖励发放开始时间
        assert_ok!(OnlineProfile::set_reward_start_era(RawOrigin::Root.into(), 0u32));
        // 设置每个Era奖励数量
        assert_ok!(OnlineProfile::set_phase_n_reward_per_era(RawOrigin::Root.into(), 0, 1_000_000u32.into()));
        assert_ok!(OnlineProfile::set_phase_n_reward_per_era(RawOrigin::Root.into(), 1, 1_000_000u32.into()));
        // 设置单卡质押上限：7_700_000_000
        assert_ok!(OnlineProfile::set_stake_usd_limit(RawOrigin::Root.into(), 7_700_000_000u64.into()));

        // stash 账户设置控制账户
        assert_ok!(OnlineProfile::set_controller(Origin::signed(stash), controller));

        // controller bond_machine
        assert_ok!(OnlineProfile::bond_machine(Origin::signed(controller), machine_id.as_bytes().to_vec()));

        let msg = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL";
        let sig = "0006e4e10234b2a6dab987ad9535b1a50829f12ba1fbb8ec2de98fa05e7f1e4da86fc36cca7262e8b698b2f52e2a0697b871a55c38f9d3d70e25cffa9eade48f";
        // submit machine signed info to confirm stash account
        assert_ok!(OnlineProfile::machine_set_stash(
            Origin::signed(controller),
            msg.as_bytes().to_vec() ,
            hex::decode(sig).unwrap() ,
            StakerCustomizeInfo{
                upload_net: 1100,
                download_net: 1101,
                longitude: 1102,
                latitude: 1103,
                ..Default::default()
            }
        ));

        // 增加三个委员会
        assert_ok!(Committee::add_committee(RawOrigin::Root.into(), one));
        assert_ok!(Committee::add_committee(RawOrigin::Root.into(), two));
        assert_ok!(Committee::add_committee(RawOrigin::Root.into(), alice));

        // 委员会提交box_pubkey
        let one_box_pubkey = hex::decode("9dccbab2d61405084eac440f877a6479bc827373b2e414e81a6170ebe5aadd12").unwrap().try_into().unwrap();
        let two_box_pubkey = hex::decode("1e71b5a83ccdeff1592062a1d4da4a272691f08e2024a1ca75a81d534a76210a").unwrap().try_into().unwrap();
        let alice_box_pubkey = hex::decode("ff3033c763f71bc51f372c1dc5095accc26880e138df84cac13c46bfd7dbd74f").unwrap().try_into().unwrap();
        assert_ok!(Committee::committee_set_box_pubkey(Origin::signed(one), one_box_pubkey));
        assert_ok!(Committee::committee_set_box_pubkey(Origin::signed(two), two_box_pubkey));
        assert_ok!(Committee::committee_set_box_pubkey(Origin::signed(alice), alice_box_pubkey));

        // 委员会处于正常状态
        assert_eq!(Committee::committee(), CommitteeList{normal: vec!(two, one, alice), ..Default::default()});

        // 订单处于正常状态
        assert_eq!(OnlineProfile::live_machines(), LiveMachine{
            machine_confirmed: vec!(machine_id.as_bytes().to_vec()),
            ..Default::default()
        });

        LeaseCommittee::distribute_machines();

        // 订单处于正常状态
        assert_eq!(OnlineProfile::live_machines(), LiveMachine{
            machine_confirmed: vec!(machine_id.as_bytes().to_vec()),
            ..Default::default()
        });

        run_to_block(10);
        assert_eq!(
            LeaseCommittee::machine_committee(machine_id.as_bytes().to_vec()),
            LCMachineCommitteeList{..Default::default()}
        );

        // 委员会分配订单

        // 委员会提交机器hash

        // 委员会提交原始信息

        //
    });
}

#[test]
fn select_committee_works() {
    // 质押--参加选举--当选
    new_test_ext().execute_with(|| {
        System::set_block_number(1);

        let alice: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Alice).into();
        let bob: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Bob).into();
        let charile: sp_core::sr25519::Public =
            sr25519::Public::from(Sr25519Keyring::Charlie).into();
        let dave: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Dave).into();
        let eve: sp_core::sr25519::Public = sr25519::Public::from(Sr25519Keyring::Eve).into();

        assert_eq!(Balances::free_balance(alice), 1000_000);

        // // 设置初始值
        // let _ = LeaseCommittee::set_min_stake(RawOrigin::Root.into(), 500_000u32.into());
        // let _ = LeaseCommittee::set_alternate_committee_limit(RawOrigin::Root.into(), 5u32);
        // let _ = LeaseCommittee::set_committee_limit(RawOrigin::Root.into(), 3u32);

        // // 参加选举，成为候选人
        // assert_ok!(LeaseCommittee::stake_for_alternate_committee(
        //     Origin::signed(alice),
        //     500_000u32.into()
        // ));
        // assert_ok!(LeaseCommittee::stake_for_alternate_committee(
        //     Origin::signed(bob),
        //     500_000u32.into()
        // ));
        // assert_ok!(LeaseCommittee::stake_for_alternate_committee(
        //     Origin::signed(charile),
        //     500_000u32.into()
        // ));
        // assert_ok!(LeaseCommittee::stake_for_alternate_committee(
        //     Origin::signed(dave),
        //     500_000u32.into()
        // ));
        // assert_ok!(LeaseCommittee::stake_for_alternate_committee(
        //     Origin::signed(eve),
        //     500_000u32.into()
        // ));

        // assert_eq!(LeaseCommittee::alternate_committee().len(), 5);
        // assert_ok!(LeaseCommittee::reelection_committee(RawOrigin::Root.into()));

        // assert_eq!(LeaseCommittee::committee().len(), 3);
        // assert_eq!(LeaseCommittee::alternate_committee().len(), 5);
    })
}

#[test]
fn book_one_machine_works() {
    new_test_ext().execute_with(|| {
        System::set_block_number(1);
    })
}

#[test]
fn bool_all_works() {
    new_test_ext().execute_with(|| {
        System::set_block_number(1);
    })
}