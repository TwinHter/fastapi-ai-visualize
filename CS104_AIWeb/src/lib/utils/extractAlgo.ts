export const extractAlgo = (algo: string) => {
    switch (algo) {
        case "tree_custom":
            return { name: "decision_tree", type: "custom" };
        case "tree_sklearn":
            return { name: "decision_tree", type: "sklearn" };
        case "lasso_custom":
            return { name: "lasso", type: "custom" };
        case "lasso_sklearn":
            return { name: "lasso", type: "sklearn" };
        case "ann_custom":
            return { name: "ann", type: "custom" };
        case "ann_sklearn":
            return { name: "ann", type: "sklearn" };
        default:
            return { name: "none", type: "none" };
    }
};
