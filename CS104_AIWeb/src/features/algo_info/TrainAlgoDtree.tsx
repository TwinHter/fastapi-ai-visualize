import { useForm } from "react-hook-form";
import { decisionTreeTrainSchema } from "../../lib/schema/trainingSchema";
import type { DecisionTreeTrainSchema } from "../../lib/schema/trainingSchema";
import { zodResolver } from "@hookform/resolvers/zod";
import { Box, Button, Typography, Paper, Stack } from "@mui/material";
import { MetricsTable } from "../../app/shared/MetricsTable";
import TextInput from "../../app/shared/TextInput";
import { useMetric } from "../../lib/hook/useMetric";
import { usePredict } from "../../lib/hook/usePredict";
import { useParams } from "react-router";

function TrainAlgoDtree() {
    const { handleSubmit, control } = useForm<DecisionTreeTrainSchema>({
        resolver: zodResolver(decisionTreeTrainSchema),
        mode: "onTouched",
    });
    const { type } = useParams();

    const onSubmit = async (data: DecisionTreeTrainSchema) => {
        const params: DecisionTreeTrainSchema = {
            max_depth: data.max_depth,
            min_samples_split: data.min_samples_split,
            random_state: data.random_state,
            min_samples_leaf: data.min_samples_leaf,
        };
        try {
            await trainDecisionTree.mutateAsync(params);
        } catch (error) {
            console.error("Error during training:", error);
        }
    };

    const { trainDecisionTree } = usePredict("decision_tree", type!);
    const { metric: algorithmMetrics } = useMetric("decision_tree", type!);

    return (
        <Box sx={{ maxWidth: 800, mx: "auto", p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom>
                Decision Tree Training ({type})
            </Typography>

            <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
                <Typography variant="subtitle1" gutterBottom>
                    Current Training Result
                </Typography>
                <MetricsTable algorithmMetrics={algorithmMetrics} />
            </Paper>

            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="subtitle1" gutterBottom>
                    Training Parameters
                </Typography>

                <Box
                    component="form"
                    onSubmit={handleSubmit(onSubmit)}
                    sx={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 2,
                        mt: 2,
                    }}
                >
                    <Stack spacing={2}>
                        <TextInput
                            control={control}
                            label="Max Depth"
                            name="max_depth"
                            defaultValue={5}
                        />
                        <TextInput
                            control={control}
                            label="Min Samples Split"
                            name="min_samples_split"
                            defaultValue={2}
                        />
                        <TextInput
                            control={control}
                            label="Random State"
                            name="random_state"
                            defaultValue={42}
                        />
                        <TextInput
                            control={control}
                            label="Min Samples Leaf"
                            name="min_samples_leaf"
                            defaultValue={1}
                        />
                    </Stack>

                    <Button
                        type="submit"
                        variant="contained"
                        color="primary"
                        sx={{ mt: 2, alignSelf: "flex-start" }}
                    >
                        Train
                    </Button>
                </Box>
            </Paper>
        </Box>
    );
}

export default TrainAlgoDtree;
