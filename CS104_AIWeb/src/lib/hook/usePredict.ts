import { useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../api";
import type { Feature } from "../types/index.t";
import type {
    AnnTrainSchema,
    DecisionTreeTrainSchema,
    LassoTrainSchema,
} from "../schema/trainingSchema";

export const usePredict = (name?: string, type?: string) => {
    const queryClient = useQueryClient();
    const predictData = useMutation({
        mutationFn: async (input: Feature) => {
            console.log(`/${name}/${type}/predict`, input);
            const response = await api.post(`/${name}/${type}/predict`, input);
            return response.data; // { value: number }
        },
    });

    const trainDecisionTree = useMutation({
        mutationFn: async (input: DecisionTreeTrainSchema) => {
            await api.post(`/${name}/${type}/train`, input);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ["metrics", name, type],
            });
        },
    });

    const trainLasso = useMutation({
        mutationFn: async (input: LassoTrainSchema) => {
            await api.post(`/${name}/${type}/train`, input);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ["metrics", name, type],
            });
        },
    });

    const trainAnn = useMutation({
        mutationFn: async (input: AnnTrainSchema) => {
            const inputData = {
                layer_sizes: input.layer_sizes
                    ?.split(",")
                    .map((x) => parseInt(x)),
                learning_rate: input.learning_rate,
                max_iter: input.max_iter,
                verbose: input.verbose,
            };
            await api.post(`/${name}/${type}/train`, inputData);
        },
        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ["metrics", name, type],
            });
        },
    });

    const trainAll = useMutation({
        mutationFn: async () => {
            await api.post("/train_all");
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["metrics"] });
        },
    });
    return {
        predictData,
        trainDecisionTree,
        trainLasso,
        trainAnn,
        trainAll,
    };
};
