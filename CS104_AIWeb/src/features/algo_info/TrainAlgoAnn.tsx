import { useForm } from "react-hook-form";
import { annTrainSchema } from "../../lib/schema/trainingSchema";
import type { AnnTrainSchema } from "../../lib/schema/trainingSchema";
import { zodResolver } from "@hookform/resolvers/zod";
import { Box, Button, Typography, Paper, Stack } from "@mui/material";
import { MetricsTable } from "../../app/shared/MetricsTable";
import TextInput from "../../app/shared/TextInput";
import { useMetric } from "../../lib/hook/useMetric";
import { usePredict } from "../../lib/hook/usePredict";
import { useParams } from "react-router";

function TrainAlgoAnn() {
    const { handleSubmit, control } = useForm<AnnTrainSchema>({
        resolver: zodResolver(annTrainSchema),
        mode: "onTouched",
    });
    const { type } = useParams();

    const onSubmit = async (data: AnnTrainSchema) => {
        const params: AnnTrainSchema = {
            layer_sizes: data.layer_sizes,
            learning_rate: data.learning_rate,
            max_iter: data.max_iter,
            verbose: data.verbose,
        };
        try {
            await trainAnn.mutateAsync(params);
        } catch (error) {
            console.error("Error during training:", error);
        }
    };

    const { trainAnn } = usePredict("ann", type!);
    const { metric: algorithmMetrics } = useMetric("ann", type!);

    return (
        <Box sx={{ maxWidth: 800, mx: "auto", p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom>
                Ann Training ({type})
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
                            label="Layer Sizes"
                            name="layer_sizes"
                            placeholder="e.g. 100,50"
                        />
                        <TextInput
                            control={control}
                            label="Learning Rate"
                            name="learning_rate"
                            defaultValue={0.1}
                        />
                        <TextInput
                            control={control}
                            label="Max Iterations"
                            name="max_iter"
                            defaultValue={10000}
                        />
                        <TextInput
                            control={control}
                            label="Verbose"
                            name="verbose"
                            defaultValue={false}
                            type="checkbox"
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

export default TrainAlgoAnn;
