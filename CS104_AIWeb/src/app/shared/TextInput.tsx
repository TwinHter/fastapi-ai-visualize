import { TextField } from "@mui/material";
import type { TextFieldProps } from "@mui/material";
import type { FieldValues, UseControllerProps } from "react-hook-form";

import { useController } from "react-hook-form";

type Props<T extends FieldValues> = {} & TextFieldProps & UseControllerProps<T>;
export default function TextInput<T extends FieldValues>(props: Props<T>) {
    const { field, fieldState } = useController<T>({ ...props });
    return (
        <TextField
            {...props}
            {...field}
            error={!!fieldState.error}
            helperText={fieldState.error?.message}
            variant="outlined"
            value={field.value ?? ""}
        />
    );
}
