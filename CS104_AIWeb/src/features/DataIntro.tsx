import {
    Box,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Stack,
    Paper,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { dataIntroduction } from "../lib/types/const-index.t";
import EdaVisualize from "./EdaVisualize";

export default function DataIntro() {
    return (
        <Box
            sx={{
                p: { xs: 2, md: 4 },
                bgcolor: "#f8f9fa",
                fontFamily: "'Roboto Slab', serif",
                minHeight: "100vh",
            }}
        >
            <Typography
                variant="h3"
                component="h1"
                gutterBottom
                sx={{
                    fontWeight: 700,
                    color: "#2c3e50",
                    textAlign: "center",
                    mb: 4,
                }}
            >
                {dataIntroduction.name}
            </Typography>

            {/** Accordion Block Component */}
            {[
                {
                    title: "Introduction",
                    content: (
                        <Typography
                            variant="body1"
                            sx={{ fontSize: "1rem", color: "#555" }}
                        >
                            {dataIntroduction.description}
                        </Typography>
                    ),
                },
                {
                    title: "Features",
                    content: (
                        <TableContainer component={Paper} elevation={2}>
                            <Table>
                                <TableHead>
                                    <TableRow sx={{ bgcolor: "#f0f4f8" }}>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Type</TableCell>
                                        <TableCell>Description</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {dataIntroduction.features.map((f) => (
                                        <TableRow key={f.name}>
                                            <TableCell>{f.name}</TableCell>
                                            <TableCell>{f.type}</TableCell>
                                            <TableCell>
                                                {f.description}
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    ),
                },
                {
                    title: "Target",
                    content: (
                        <TableContainer component={Paper} elevation={2}>
                            <Table>
                                <TableHead>
                                    <TableRow sx={{ bgcolor: "#f0f4f8" }}>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Type</TableCell>
                                        <TableCell>Description</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    <TableRow
                                        key={dataIntroduction.target.name}
                                    >
                                        <TableCell>
                                            {dataIntroduction.target.name}
                                        </TableCell>
                                        <TableCell>
                                            {dataIntroduction.target.type}
                                        </TableCell>
                                        <TableCell>
                                            {
                                                dataIntroduction.target
                                                    .description
                                            }
                                        </TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>
                        </TableContainer>
                    ),
                },
                {
                    title: "Statistics",
                    content: (
                        <Stack spacing={2} sx={{ mt: 1 }}>
                            {dataIntroduction.statistics.map((s) => (
                                <Stack
                                    direction="row"
                                    spacing={2}
                                    key={s.name}
                                    alignItems="center"
                                >
                                    <Typography sx={{ minWidth: 120 }}>
                                        {s.name}:
                                    </Typography>
                                    <Chip label={s.value} color="primary" />
                                </Stack>
                            ))}
                        </Stack>
                    ),
                },
                {
                    title: "Metrics",
                    content: (
                        <Stack direction="row" spacing={1} flexWrap="wrap">
                            {dataIntroduction.metrics.map((m) => (
                                <Chip
                                    label={m.name}
                                    key={m.name}
                                    color="secondary"
                                    variant="outlined"
                                />
                            ))}
                        </Stack>
                    ),
                },
                {
                    title: "Sample",
                    content: (
                        <TableContainer component={Paper} elevation={2}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow sx={{ bgcolor: "#f0f4f8" }}>
                                        {Object.keys(
                                            dataIntroduction.sample[0]
                                        ).map((key) => (
                                            <TableCell key={key}>
                                                {key}
                                            </TableCell>
                                        ))}
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {dataIntroduction.sample.map((s, index) => (
                                        <TableRow key={index}>
                                            {Object.values(s).map(
                                                (value, i) => (
                                                    <TableCell key={i}>
                                                        {value}
                                                    </TableCell>
                                                )
                                            )}
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    ),
                },
                {
                    title: "EDA",
                    content: <EdaVisualize />,
                },
                {
                    title: "Preprocessing",
                    content: (
                        <Typography variant="body1">
                            Hello, my name is <strong>Nguyen</strong>.
                        </Typography>
                    ),
                },
            ].map((section, index) => (
                <Accordion
                    key={section.title}
                    defaultExpanded={index === 0}
                    sx={{
                        mb: 2,
                        border: "1px solid #ddd",
                        borderRadius: 1,
                        boxShadow: "0 2px 6px rgba(0,0,0,0.08)",
                        "&::before": {
                            display: "none",
                        },
                    }}
                >
                    <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        sx={{
                            bgcolor: "#00e5ff",
                            fontWeight: 600,
                        }}
                    >
                        <Typography
                            sx={{
                                fontSize: "1.1rem",
                                fontWeight: 600,
                                color: "#333",
                            }}
                        >
                            {section.title}
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>{section.content}</AccordionDetails>
                </Accordion>
            ))}
        </Box>
    );
}
