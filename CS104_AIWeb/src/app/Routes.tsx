import { createBrowserRouter } from "react-router";
import HomePage from "../features/HomePage";
import DataIntro from "../features/DataIntro";
import App from "./layout/App";
import CompareAlgo from "../features/test_algo/CompareAlgo";
import PredictForm from "../features/test_algo/PredictForm";
import Algorithm from "../features/algo_info/Algorithm";
import TrainAlgoDtree from "../features/algo_info/TrainAlgoDtree";
import TrainAlgoLasso from "../features/algo_info/TrainAlgoLasso";
import TrainAlgoAnn from "../features/algo_info/TrainAlgoAnn";
import PlotImage from "../features/PlotImage";
const router = createBrowserRouter([
    {
        path: "/",
        element: <App />,
        children: [
            { path: "", element: <HomePage /> },
            { path: "data-intro", element: <DataIntro /> },
            { path: "compare-algo", element: <CompareAlgo /> },
            { path: "predict", element: <PredictForm /> },
            { path: "algorithm", element: <Algorithm /> },
            { path: `train/decision_tree/:type`, element: <TrainAlgoDtree /> },
            { path: `train/lasso/:type`, element: <TrainAlgoLasso /> },
            { path: `train/ann/:type`, element: <TrainAlgoAnn /> },
            { path: `image`, element: <PlotImage /> },
        ],
    },
]);
export default router;
