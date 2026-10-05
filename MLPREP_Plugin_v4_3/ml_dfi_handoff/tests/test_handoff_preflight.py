from handoff_preflight import (
    find_machine_specific_text,
    is_absolute_path,
    package_exclusion_reason,
)


def test_machine_specific_text_detection_avoids_colon_false_positive():
    windows_path = "C" + ":" + "\\Users\\analyst\\project"
    posix_home = "/" + "home" + "/analyst/project"
    legacy_workspace = "Codes" + "_V1"

    assert find_machine_specific_text(f'path = "{windows_path}"')
    assert find_machine_specific_text(f'path = "{posix_home}"')
    assert find_machine_specific_text(legacy_workspace)
    assert not find_machine_specific_text('"errors:\\n- first failure"')


def test_absolute_path_detection_is_host_independent():
    windows_path = "D" + ":" + "\\project\\config.json"
    posix_path = "/" + "srv/project/config.json"

    assert is_absolute_path(windows_path)
    assert is_absolute_path(posix_path)
    assert not is_absolute_path("../config/default.json")


def test_package_policy_excludes_data_models_logs_and_local_settings():
    assert package_exclusion_reason("Sample Data/Input/private_inventory.shp")
    assert package_exclusion_reason("outputs/model.pkl")
    assert package_exclusion_reason("run/processing.log")
    assert package_exclusion_reason("config/local_config.json")
    assert package_exclusion_reason(".venv-local/Lib/site-packages/example.py")
    assert package_exclusion_reason("configs/archive/experiments/test.json")


def test_package_policy_keeps_source_and_reviewed_step4_executable():
    assert package_exclusion_reason(
        "steps/step3_train_ml_dfi_model/train.py"
    ) is None
    assert package_exclusion_reason(
        "steps/step5_post_depositional_spread_PDS/step5.md"
    ) is None
    assert package_exclusion_reason(
        "steps/step6_landslide_damming_potential_LDP/step6.md"
    ) is None
    assert (
        package_exclusion_reason(
            "steps/step4_deposition_zone/cpp_mpi_port/build_manual/"
            "step4_deposition_zone_mpi.exe"
        )
        is None
    )
