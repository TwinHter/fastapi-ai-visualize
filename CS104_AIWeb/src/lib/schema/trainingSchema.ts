import { z } from "zod";
export const decisionTreeTrainSchema = z.object({
    max_depth: z.coerce
        .number()
        .int()
        .positive("Max depth must be positive.")
        .optional(),
    min_samples_split: z.coerce
        .number()
        .int()
        .positive("Min samples split must be positive.")
        .optional(),
    random_state: z.coerce.number().int().optional(),
    min_samples_leaf: z.coerce.number().int().optional(),
});

export const lassoTrainSchema = z.object({
    alpha: z.coerce.number().positive("Alpha must be positive.").optional(),
    fit_intercept: z.coerce.boolean().default(true).optional(),
    max_iter: z.coerce
        .number()
        .int()
        .positive("Max iterations must be positive.")
        .optional(),
});

export const annTrainSchema = z.object({
    layer_sizes: z.string().optional(),
    learning_rate: z.coerce
        .number()
        .positive("Learning rate must be positive.")
        .optional(),
    max_iter: z.coerce
        .number()
        .int()
        .positive("Number of iterations must be positive.")
        .optional(),
    verbose: z.coerce.boolean().default(false).optional(),
});
export type AnnTrainSchema = z.infer<typeof annTrainSchema>;
export type DecisionTreeTrainSchema = z.infer<typeof decisionTreeTrainSchema>;
export type LassoTrainSchema = z.infer<typeof lassoTrainSchema>;
