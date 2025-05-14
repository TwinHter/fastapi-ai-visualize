import { z } from "zod";
import { countryList } from "../types/const-index.t";

export const testSchema = z.object({
    Area: z.enum(countryList),
    Item: z.string(),
    Year: z.coerce.number().int(),
    average_rain_fall_mm_per_year: z.coerce
        .number()
        .positive("Average rainfall must be positive."),
    pesticides_tonnes: z.coerce
        .number()
        .positive("Pesticides tonnes must be positive."),
    avg_temp: z.coerce
        .number()
        .positive("Average temperature must be positive."),
});
export type TestSchema = z.infer<typeof testSchema>;
