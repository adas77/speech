export function strToBlob(data: string): Blob {
    const blob = new Blob([data], {
        type: 'text/plain'
    })
    return blob
}

