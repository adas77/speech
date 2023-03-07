export function str_to_blob(data: string): Blob {
    const blob = new Blob([data], {
        type: 'text/plain'
    })
    return blob
} 