from fastapi import APIRouter
import schemas.predictSchema as schemas
import models.decision_tree_custom as decision_tree_custom
import models.decision_tree_sklearn as decision_tree_sklearn
import models.lasso_custom as lasso_custom
import models.lasso_sklearn as lasso_sklearn
import models.ann_sklearn as ann_sklearn
import models.ann_custom as ann_custom

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
@router.post("/{name}/{type}/predict", response_model=schemas.PredictResponse)
def algo_predict(name: str, type: str, req: schemas.PredictRequest):
    print(req)
    func = algorithm_map[name][type]
    prediction = func.predict(req)
    return {"hg_ha_yield": prediction}
    
@router.post("/lasso/{type}/train")
def lasso_train(type: str, req: schemas.LassoParam):
    if type == "sklearn":
        lasso_sklearn.train(req)
    else:
        lasso_custom.train(req)
    return {"message": "Model {name} + {type} trained successfully"}

@router.post("/decision_tree/{type}/train")
def decision_tree_train(type: str, req: schemas.DecisionTreeParam):
    if type == "sklearn":
        decision_tree_sklearn.train(req)
    else:
        decision_tree_custom.train(req)
    return {"message": "Model {name} + {type} trained successfully"}

@router.post("/ann/{type}/train")
def ann_train(type: str, req: schemas.AnnParam):
    if type == "sklearn":
        ann_sklearn.train(req)
    else:
        ann_custom.train(req)
    return {"message": "Model {name} + {type} trained successfully"}

@router.post("/train_all")
def train_all():
    lasso_sklearn.train()
    print("Lasso Sklearn trained")
    lasso_custom.train()
    print("Lasso Custom trained")
    decision_tree_sklearn.train()
    print("Decision Tree Sklearn trained")
    decision_tree_custom.train()
    print("Decision Tree Custom trained")
    ann_sklearn.train()
    print("ANN Sklearn trained")
    ann_custom.train()
    print("ANN Custom trained")
    return {"message": "All models trained successfully"}
