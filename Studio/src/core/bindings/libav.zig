pub const video = @cImport({
    @cInclude("libavfilter/avfilter.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavutil/frame.h");
    @cInclude("libavutil/hwcontext.h");
    @cInclude("libavutil/hwcontext_cuda.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libavutil/pixdesc.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libavfilter/buffersink.h");
    @cInclude("libavfilter/buffersrc.h");
    @cInclude("libavutil/opt.h");
});
