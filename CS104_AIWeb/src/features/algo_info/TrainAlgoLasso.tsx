import { useForm } from "react-hook-form";
import { lassoTrainSchema } from "../../lib/schema/trainingSchema";
import type { LassoTrainSchema } from "../../lib/schema/trainingSchema";
import { zodResolver } from "@hookform/resolvers/zod";
import { Box, Button, Typography, Paper, Stack } from "@mui/material";
import { MetricsTable } from "../../app/shared/MetricsTable";
import TextInput from "../../app/shared/TextInput";
import { useMetric } from "../../lib/hook/useMetric";
import { usePredict } from "../../lib/hook/usePredict";
import { useParams } from "react-router";

function TrainAlgoLasso() {
    const { handleSubmit, control } = useForm<LassoTrainSchema>({
        resolver: zodResolver(lassoTrainSchema),
        mode: "onTouched",
    });
    const { type } = useParams();

    const onSubmit = async (data: LassoTrainSchema) => {
        const params: LassoTrainSchema = {
            max_iter: data.max_iter,
            alpha: data.alpha,
            fit_intercept: data.fit_intercept,
        };
        try {
            await trainLasso.mutateAsync(params);
        } catch (error) {
            console.error("Error during training:", error);
        }
    };

    const { trainLasso } = usePredict("lasso", type!);
    const { metric: algorithmMetrics } = useMetric("lasso", type!);

    return (
        <Box sx={{ maxWidth: 800, mx: "auto", p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom>
                Lasso Training ({type})
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
                            label="Max Iterations"
                            name="max_iter"
                            defaultValue={100}
                        />
                        <TextInput
                            control={control}
                            label="alpha"
                            name="alpha"
                            defaultValue={1}
                        />
                        <TextInput
                            control={control}
                            label="Fit Intercept"
                            name="fit_intercept"
                            defaultValue={true}
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

export default TrainAlgoLasso;
