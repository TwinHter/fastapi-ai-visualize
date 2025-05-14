import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import type { SelectChangeEvent } from "@mui/material";
import React from "react";

export type SelectOption = {
    label: string;
    value: string;
};

type SelectInputProps = {
    label: string;
    value: string;
    onChange: (event: SelectChangeEvent) => void;
    options: SelectOption[];
    id?: string;
    minWidth?: number;
};

const SelectInput: React.FC<SelectInputProps> = ({
    label,
    value,
    onChange,
    options,
    id = "shared-select",
    minWidth = 240,
}) => {
    return (
        <FormControl sx={{ minWidth, my: 2 }}>
            <InputLabel id={`${id}-label`}>{label}</InputLabel>
            <Select
                labelId={`${id}-label`}
                id={id}
                value={value}
                label={label}
                onChange={onChange}
            >
                {options.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
};

export default SelectInput;
