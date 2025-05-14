import {
    TableContainer,
    Paper,
    Table,
    TableHead,
    TableRow,
    TableCell,
    TableBody,
} from "@mui/material";
import type { Metric } from "../../lib/types/index.t";

type Props = {
    algorithmMetrics: Metric | undefined;
};
export const MetricsTable = ({ algorithmMetrics }: Props) => {
    return !algorithmMetrics ? (
        <TableContainer component={Paper} sx={{ mt: 2 }}>
            <Table>
                <TableBody>
                    <TableRow>
                        <TableCell colSpan={2} align="center">
                            No metrics available
                        </TableCell>
                    </TableRow>
                </TableBody>
            </Table>
        </TableContainer>
    ) : (
        <TableContainer component={Paper} sx={{ mt: 2 }}>
            <Table>
                <TableHead>
                    <TableRow>
                        <TableCell>Metric</TableCell>
                        <TableCell>Value</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {Object.entries(algorithmMetrics).map(([metric, value]) => (
                        <TableRow key={metric}>
                            <TableCell>{metric}</TableCell>
                            <TableCell>{value}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};
