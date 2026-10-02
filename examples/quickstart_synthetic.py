"""Fast executable demonstration of the RAVEL core."""

from pathlib import Path

from ravel import (
    fit_selected_shared_route,
    fit_session_model,
    paper_config,
    save_config,
    save_physics_json,
)
from ravel.synthetic import make_dynamic_route_cohort
from ravel.workflows import score_shared_route


def main() -> None:
    config = paper_config()
    sessions, true_atoms = make_dynamic_route_cohort(3, config=config)

    first_static = fit_session_model(
        sessions[0], true_atoms, "active", dynamic=False, config=config
    )
    first_dynamic = fit_session_model(
        sessions[0], true_atoms, "active", dynamic=True, config=config
    )
    print("static train MSE:", first_static.train_mse)
    print("dynamic train MSE:", first_dynamic.train_mse)

    output = Path("synthetic_ravel_output")
    shared = fit_selected_shared_route(
        "synthetic_X_to_Y",
        sessions,
        true_atoms,
        context="active",
        checkpoint_dir=output / "checkpoints",
        config=config,
    )
    scores = score_shared_route(
        sessions, shared, context="active", config=config
    )
    output.mkdir(exist_ok=True)
    scores.to_csv(output / "test_scores.csv", index=False)
    save_physics_json(
        shared.route_name,
        shared.atoms,
        output / "shared_physics.json",
        metadata={"selected_strategy": shared.selected_strategy},
    )
    save_config(config, output / "config.json")
    print(scores[["session_id", "gain_over_nuisance", "r2"]])
    print("fitted atoms:", [atom.label() for atom in shared.atoms])


if __name__ == "__main__":
    main()

