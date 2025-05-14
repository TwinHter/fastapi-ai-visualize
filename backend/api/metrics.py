from fastapi import APIRouter
import models.decision_tree_custom as decision_tree_custom
import models.decision_tree_sklearn as decision_tree_sklearn
import models.lasso_custom as lasso_custom
import models.lasso_sklearn as lasso_sklearn
import models.ann_sklearn as ann_sklearn
import models.ann_custom as ann_custom
import schemas.predictSchema as schemas

router = APIRouter()
algorithm_map = {
    "lasso": {
        "sklearn": lasso_sklearn,
        "custom": lasso_custom,
    },
    "decision_tree": {
        "sklearn": decision_tree_sklearn,
        "custom": decision_tree_custom,
    },
    "ann": {
        "sklearn": ann_sklearn,
        "custom": ann_custom,
    },
}

@router.get("/{name}/{type}/metrics", response_model=dict)
def api_metrics(name: str, type: str):
    func = algorithm_map[name][type]
    result = func.get_metrics_test()  # Gọi hàm tương ứng
    return result
