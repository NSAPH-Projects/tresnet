import torch


def criterion(
    model_output,
    true_outcome,
    target_reg_coeff=None,
    alpha=0.5,
    beta=1.0,
    epsilon=1e-6,
    use_targeted_regularizer=True,
):

    if use_targeted_regularizer:
        if target_reg_coeff == None:
            raise ValueError(
                "target_reg_coeff should be defined here. It is obtained by training and nn.Module instance of TargetedRegularizerCoeff alongside the model"
            )

        loss_1 = prediction_and_likelihood_loss(
            model_output, true_outcome, alpha=0.5, epsilon=1e-6
        )
        loss_2 = functional_targeted_reg_loss(
            model_output, target_reg_coeff, true_outcome, beta=1.0, epsilon=1e-6
        )
        return {"total_loss": loss_1 + loss_2, "targeted_reg_loss": loss_2}

    else:
        loss_1 = prediction_and_likelihood_loss(
            model_output, true_outcome, alpha=0.5, epsilon=1e-6
        )

        return {"total_loss": loss_1, "targeted_reg_loss": None}


def prediction_and_likelihood_loss(model_output, true_outcome, alpha=0.5, epsilon=1e-6):

    return (
        (model_output["predicted_outcome"].squeeze() - true_outcome.squeeze()) ** 2
    ).mean() - alpha * torch.log(model_output["prob_score"] + epsilon).mean()


def functional_targeted_reg_loss(
    model_output, target_reg_coeff, true_outcome, beta=1.0, epsilon=1e-6
):

    return (
        beta
        * (
            (
                true_outcome.squeeze()
                - model_output["predicted_outcome"].squeeze()
                - (
                    target_reg_coeff.squeeze()
                    / (model_output["prob_score"].squeeze() + epsilon)
                )
            )
            ** 2
        ).mean()
    )
