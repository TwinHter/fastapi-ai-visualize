import api from "../api";
import type { Metric } from "../types/index.t";
import { useQuery } from "@tanstack/react-query";

export const useMetric = (name: string, type: string) => {
    const { data: metric } = useQuery<Metric>({
        queryKey: ["metrics", name, type],
        queryFn: async () => {
            const response = await api.get(`/${name}/${type}/metrics`);
            return response.data;
        },
        enabled: !!type && !!name,
        refetchOnWindowFocus: false,
    });

    return { metric };
};
